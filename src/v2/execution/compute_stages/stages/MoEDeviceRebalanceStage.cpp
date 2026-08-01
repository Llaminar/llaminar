/**
 * @file MoEDeviceRebalanceStage.cpp
 * @brief Implementation of graph-capturable MoE device rebalance stage.
 */

#include "MoEDeviceRebalanceStage.h"

#include "../../../backends/BackendManager.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../backends/IWorkerGPUContext.h"
#include "../../../collective/ILocalTPContext.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../execution/moe/MoERuntimeTable.h"
#include "../../../kernels/IMoEKernel.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace llaminar2
{
    namespace
    {
        using KernelFactory = llaminar::v2::kernels::KernelFactory;

        constexpr const char *kDefaultStageName = "moe_device_rebalance";

        std::string suffixFor(const std::string &stage_name)
        {
            return stage_name.empty() ? std::string(kDefaultStageName) : stage_name;
        }

        const char *phaseName(DeviceMoERebalanceStagePhase phase)
        {
            switch (phase)
            {
            case DeviceMoERebalanceStagePhase::PlanCopyApply:
                return "plan_copy_apply";
            case DeviceMoERebalanceStagePhase::CollectState:
                return "collect_state";
            case DeviceMoERebalanceStagePhase::PlanAndCopy:
                return "plan_and_copy";
            case DeviceMoERebalanceStagePhase::PlanAndCopyAfterSideband:
                return "plan_and_copy_after_sideband";
            case DeviceMoERebalanceStagePhase::PackCollectivePayloadAfterSideband:
                return "pack_collective_payload_after_sideband";
            case DeviceMoERebalanceStagePhase::UnpackCollectivePayloadAfterSideband:
                return "unpack_collective_payload_after_sideband";
            case DeviceMoERebalanceStagePhase::Apply:
                return "apply";
            case DeviceMoERebalanceStagePhase::JoinTransfer:
                return "join_transfer";
            default:
                return "unknown";
            }
        }

        const char *boolString(bool value)
        {
            return value ? "true" : "false";
        }

    } // namespace

    std::string MoEDeviceRebalanceStage::workspaceBufferName(
        const char *base_name,
        const std::string &workspace_name)
    {
        if (!base_name || !*base_name)
            throw std::invalid_argument("MoEDeviceRebalanceStage workspace buffer base name is required");
        return std::string(base_name) + "_" + suffixFor(workspace_name);
    }

    DeviceMoERebalanceTransferState::~DeviceMoERebalanceTransferState()
    {
        release();
    }

    bool DeviceMoERebalanceTransferState::ensure(
        DeviceId device,
        const std::string &name_suffix)
    {
        if (transfer_stream_ && compute_ready_event_ && transfer_done_event_)
            return true;

        IBackend *backend = getBackendFor(device);
        if (!backend)
        {
            LOG_ERROR("[DeviceMoERebalanceTransferState] Could not resolve backend for "
                      << device.to_string());
            return false;
        }

        int device_ordinal = -1;
        try
        {
            device_ordinal = device.gpu_ordinal();
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[DeviceMoERebalanceTransferState] Could not resolve GPU ordinal for "
                      << device.to_string() << ": " << e.what());
            return false;
        }

        IWorkerGPUContext *gpu_ctx = nullptr;
        try
        {
            gpu_ctx = &GPUDeviceContextPool::instance().getContext(device);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[DeviceMoERebalanceTransferState] Could not resolve worker GPU context for "
                      << device.to_string() << ": " << e.what());
            return false;
        }

        transfer_stream_ = gpu_ctx->getOrCreateAuxiliaryStream(
            "moe_device_rebalance_transfer:" + suffixFor(name_suffix));
        if (!transfer_stream_)
        {
            LOG_ERROR("[DeviceMoERebalanceTransferState] Failed to create context-owned transfer stream for "
                      << device.to_string());
            return false;
        }

        event_backend_ = backend;
        event_device_ordinal_ = device_ordinal;
        if (!compute_ready_event_)
            compute_ready_event_ = backend->createEvent(device_ordinal);
        if (!transfer_done_event_)
            transfer_done_event_ = backend->createEvent(device_ordinal);

        if (!compute_ready_event_ || !transfer_done_event_)
        {
            LOG_ERROR("[DeviceMoERebalanceTransferState] Failed to create transfer stream ordering events for "
                      << device.to_string());
            release();
            return false;
        }

        return true;
    }

    bool DeviceMoERebalanceTransferState::prepareForCapture(
        DeviceId device,
        const std::string &name_suffix,
        void *capture_stream)
    {
        if (!capture_stream)
        {
            LOG_ERROR("[DeviceMoERebalanceTransferState] Graph launch preparation requires an explicit capture stream"
                      << " device=" << device.to_string()
                      << " lane=" << suffixFor(name_suffix));
            return false;
        }
        if (!ensure(device, name_suffix))
            return false;

        IWorkerGPUContext *gpu_ctx = nullptr;
        try
        {
            gpu_ctx = &GPUDeviceContextPool::instance().getContext(device);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[DeviceMoERebalanceTransferState] Failed to resolve GPU context for pre-capture transfer-lane fence"
                      << " device=" << device.to_string()
                      << " lane=" << suffixFor(name_suffix)
                      << ": " << e.what());
            return false;
        }
        if (!gpu_ctx || !transfer_stream_ || !transfer_done_event_)
        {
            LOG_ERROR("[DeviceMoERebalanceTransferState] Pre-capture transfer-lane fence requires initialized resources"
                      << " device=" << device.to_string()
                      << " lane=" << suffixFor(name_suffix));
            return false;
        }

        /*
         * Publish all previously submitted auxiliary work before capture begins.
         * This edge is outside the new graph. The stage's ordinary
         * computeReady/transferDone event pair then becomes part of the captured
         * graph and orders each payload transaction during replay.
         */
        if (!gpu_ctx->recordEventChecked(transfer_done_event_, transfer_stream_) ||
            !gpu_ctx->waitEventChecked(transfer_done_event_, capture_stream))
        {
            LOG_ERROR("[DeviceMoERebalanceTransferState] Failed to queue pre-capture transfer-lane event fence"
                      << " device=" << device.to_string()
                      << " lane=" << suffixFor(name_suffix));
            return false;
        }
        return true;
    }

    void DeviceMoERebalanceTransferState::release()
    {
        if (event_backend_ && event_device_ordinal_ >= 0)
        {
            if (compute_ready_event_)
                event_backend_->destroyEvent(
                    compute_ready_event_,
                    event_device_ordinal_);
            if (transfer_done_event_)
                event_backend_->destroyEvent(
                    transfer_done_event_,
                    event_device_ordinal_);
        }
        compute_ready_event_ = nullptr;
        transfer_done_event_ = nullptr;
        event_backend_ = nullptr;
        event_device_ordinal_ = -1;
        /*
         * transfer_stream_ is context-owned via IWorkerGPUContext. It remains
         * valid until the context resets auxiliary streams or shuts down.
         */
        transfer_stream_ = nullptr;
    }

    MoEDeviceRebalanceStage::MoEDeviceRebalanceStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    MoEDeviceRebalanceStage::~MoEDeviceRebalanceStage() = default;

    void MoEDeviceRebalanceStage::invalidateKernelDynamicState()
    {
        if (!owned_moe_kernel_)
            return;

        owned_moe_kernel_->resetDynamicState();
        owned_moe_kernel_->clearGPUStreamBinding();
    }

    std::string MoEDeviceRebalanceStage::localHistogramBufferName() const
    {
        return std::string(WS_LOCAL_HISTOGRAM) + "_" + workspaceSuffix();
    }

    std::string MoEDeviceRebalanceStage::gatheredHistogramBufferName() const
    {
        return std::string(WS_GATHERED_HISTOGRAM) + "_" + workspaceSuffix();
    }

    std::string MoEDeviceRebalanceStage::transferPlanBufferName() const
    {
        return std::string(WS_TRANSFER_PLAN) + "_" + workspaceSuffix();
    }

    std::string MoEDeviceRebalanceStage::transferPlanCountBufferName() const
    {
        return std::string(WS_TRANSFER_PLAN_COUNT) + "_" + workspaceSuffix();
    }

    std::string MoEDeviceRebalanceStage::commandHeaderBufferName() const
    {
        return std::string(WS_COMMAND_HEADER) + "_" + workspaceSuffix();
    }

    std::string MoEDeviceRebalanceStage::controllerStateBufferName() const
    {
        return std::string(WS_CONTROLLER_STATE) + "_" + workspaceSuffix();
    }

    std::string MoEDeviceRebalanceStage::gatheredTransferPlanBufferName() const
    {
        return std::string(WS_GATHERED_TRANSFER_PLAN) + "_" + workspaceSuffix();
    }

    std::string MoEDeviceRebalanceStage::gatheredCommandHeaderBufferName() const
    {
        return std::string(WS_GATHERED_COMMAND_HEADER) + "_" + workspaceSuffix();
    }

    std::string MoEDeviceRebalanceStage::waveStateBufferName() const
    {
        return std::string(WS_WAVE_STATE) + "_" + workspaceSuffix();
    }

    std::string MoEDeviceRebalanceStage::gatheredWaveStateBufferName() const
    {
        return std::string(WS_GATHERED_WAVE_STATE) + "_" + workspaceSuffix();
    }

    std::string MoEDeviceRebalanceStage::statusBufferName() const
    {
        return std::string(WS_STATUS) + "_" + workspaceSuffix();
    }

    std::string MoEDeviceRebalanceStage::localDirectoryBufferName() const
    {
        return std::string(WS_LOCAL_DIRECTORY) + "_" + workspaceSuffix();
    }

    std::string MoEDeviceRebalanceStage::localSourceDescriptorsBufferName() const
    {
        return std::string(WS_LOCAL_SOURCE_DESCRIPTORS) + "_" + workspaceSuffix();
    }

    std::string MoEDeviceRebalanceStage::localTransferPayloadBufferName() const
    {
        return std::string(WS_LOCAL_TRANSFER_PAYLOAD) + "_" + workspaceSuffix();
    }

    std::string MoEDeviceRebalanceStage::gatheredTransferPayloadBufferName() const
    {
        return std::string(WS_GATHERED_TRANSFER_PAYLOAD) + "_" + workspaceSuffix();
    }

    std::string MoEDeviceRebalanceStage::copyStatusBufferName() const
    {
        return std::string(WS_COPY_STATUS) + "_" + workspaceSuffix();
    }

    std::string MoEDeviceRebalanceStage::gatheredCopyStatusBufferName() const
    {
        return std::string(WS_GATHERED_COPY_STATUS) + "_" + workspaceSuffix();
    }

    std::string MoEDeviceRebalanceStage::applyStatusBufferName() const
    {
        return std::string(WS_APPLY_STATUS) + "_" + workspaceSuffix();
    }

    size_t MoEDeviceRebalanceStage::localHistogramEntries() const
    {
        return histogramLayerCount() *
               static_cast<size_t>(params_.config.num_experts);
    }

    size_t MoEDeviceRebalanceStage::gatheredHistogramEntries() const
    {
        return localHistogramEntries() *
               static_cast<size_t>(params_.config.participant_count);
    }

    size_t MoEDeviceRebalanceStage::localDirectoryEntries() const
    {
        return static_cast<size_t>(params_.config.num_layers) *
               static_cast<size_t>(params_.config.num_experts);
    }

    size_t MoEDeviceRebalanceStage::localSourceDescriptorEntries() const
    {
        return static_cast<size_t>(params_.config.participant_count) *
               commandBufferCount() *
               transferPlanCapacity();
    }

    size_t MoEDeviceRebalanceStage::histogramLayerCount() const
    {
        const uint32_t window_count =
            params_.config.layer_window_count == 0u
                ? params_.config.num_layers
                : std::min(params_.config.layer_window_count,
                           params_.config.num_layers);
        const uint32_t wave_count =
            params_.config.layer_wave_count == 0u
                ? window_count
                : std::min(params_.config.layer_wave_count, window_count);
        return static_cast<size_t>(std::max<uint32_t>(1u, wave_count));
    }

    size_t MoEDeviceRebalanceStage::transferPlanEntries() const
    {
        return static_cast<size_t>(
            deviceMoERebalanceCommandPlanCapacity(params_.config, params_.transfer_mode));
    }

    size_t MoEDeviceRebalanceStage::transferPlanCapacity() const
    {
        return transferPlanEntries();
    }

    size_t MoEDeviceRebalanceStage::payloadSlotCapacity() const
    {
        if (!usesCollectivePayloadLane())
            return 0;
        const size_t captured_slot_capacity =
            params_.collective_payload_slot_capacity == 0
                ? static_cast<size_t>(params_.local_transfer_slot_count)
                : std::min<size_t>(
                      static_cast<size_t>(params_.collective_payload_slot_capacity),
                      static_cast<size_t>(params_.local_transfer_slot_count));
        return std::min<size_t>(
            transferPlanCapacity(),
            captured_slot_capacity);
    }

    size_t MoEDeviceRebalanceStage::commandBufferCount() const
    {
        return usesTransferSlotApply() ? 2u : 1u;
    }

    size_t MoEDeviceRebalanceStage::collectivePayloadSlotCount() const
    {
        const size_t slot_capacity = payloadSlotCapacity();
        if (usesCompactTransferSlots())
            return slot_capacity;
        return slot_capacity * static_cast<size_t>(params_.config.participant_count);
    }

    size_t MoEDeviceRebalanceStage::collectivePayloadLocalBytes() const
    {
        return collectivePayloadSlotCount() *
               static_cast<size_t>(params_.collective_payload_slot_bytes);
    }

    size_t MoEDeviceRebalanceStage::collectivePayloadGatheredBytes() const
    {
        return collectivePayloadLocalBytes() *
               static_cast<size_t>(params_.config.participant_count);
    }

    bool MoEDeviceRebalanceStage::usesTransferSlotApply() const
    {
        return deviceMoERebalanceModeUsesTransferSlots(params_.transfer_mode) &&
               params_.local_transfer_slots != nullptr &&
               params_.local_transfer_slot_count > 0;
    }

    bool MoEDeviceRebalanceStage::usesCompactTransferSlots() const
    {
        return usesTransferSlotApply() &&
               params_.transfer_mode == DeviceMoERebalanceTransferMode::CompactTransferSlots;
    }

    bool MoEDeviceRebalanceStage::usesFixedPayloadTransfer() const
    {
        return usesTransferSlotApply() &&
               deviceMoERebalanceModeMovesFixedPayloadCapacity(params_.transfer_mode);
    }

    bool MoEDeviceRebalanceStage::usesCollectivePayloadLane() const
    {
        return usesTransferSlotApply() &&
               deviceMoERebalanceModeUsesCollectivePayloadLane(params_.transfer_mode);
    }

    bool MoEDeviceRebalanceStage::usesReadyWaveApply() const
    {
        return usesTransferSlotApply() ||
               hasDeviceMoERebalanceFlag(
                   params_.config.flags,
                   DeviceMoERebalanceFlags::DeferRuntimeApply);
    }

    bool MoEDeviceRebalanceStage::collectsState() const
    {
        return params_.phase == DeviceMoERebalanceStagePhase::PlanCopyApply ||
               params_.phase == DeviceMoERebalanceStagePhase::CollectState ||
               params_.phase == DeviceMoERebalanceStagePhase::PlanAndCopy;
    }

    bool MoEDeviceRebalanceStage::gathersStateInline() const
    {
        return params_.phase == DeviceMoERebalanceStagePhase::PlanCopyApply ||
               params_.phase == DeviceMoERebalanceStagePhase::PlanAndCopy;
    }

    bool MoEDeviceRebalanceStage::runsController() const
    {
        return params_.phase == DeviceMoERebalanceStagePhase::PlanCopyApply ||
               params_.phase == DeviceMoERebalanceStagePhase::PlanAndCopy ||
               params_.phase == DeviceMoERebalanceStagePhase::PlanAndCopyAfterSideband;
    }

    bool MoEDeviceRebalanceStage::ownsRequestTransactionState() const noexcept
    {
        return runsController();
    }

    bool MoEDeviceRebalanceStage::resetRequestTransactionStateOnStream(void *stream)
    {
        if (!ownsRequestTransactionState())
        {
            LOG_ERROR("[MoEDeviceRebalanceStage] Request-transaction reset was invoked on a non-owner phase"
                      << " stage=" << suffixFor(params_.stage_name)
                      << " phase=" << phaseName(params_.phase));
            return false;
        }
        if (!stream)
        {
            LOG_ERROR("[MoEDeviceRebalanceStage] Request-transaction reset requires the exact non-null reset stream"
                      << " stage=" << suffixFor(params_.stage_name));
            return false;
        }
        if (!bound_workspace_)
        {
            LOG_ERROR("[MoEDeviceRebalanceStage] Request-transaction reset requires bound persistent workspace"
                      << " stage=" << suffixFor(params_.stage_name));
            return false;
        }
        if (!validateCommon("MoEDeviceRebalanceStage::resetRequestTransactionStateOnStream"))
            return false;

        auto *controller_state =
            static_cast<DeviceMoERebalanceGraphControllerState *>(
                bound_workspace_->getBuffer(controllerStateBufferName()));
        auto *command_headers =
            static_cast<DeviceMoERebalanceCommandBufferHeader *>(
                bound_workspace_->getBuffer(commandHeaderBufferName()));
        auto *wave_states =
            static_cast<DeviceMoERebalanceWaveState *>(
                bound_workspace_->getBuffer(waveStateBufferName()));
        auto *plan_counts =
            static_cast<uint32_t *>(
                bound_workspace_->getBuffer(transferPlanCountBufferName()));
        if (!controller_state || !command_headers || !wave_states || !plan_counts)
        {
            LOG_ERROR("[MoEDeviceRebalanceStage] Request-transaction workspace is incomplete"
                      << " controller=" << static_cast<void *>(controller_state)
                      << " headers=" << static_cast<void *>(command_headers)
                      << " waves=" << static_cast<void *>(wave_states)
                      << " plan_counts=" << static_cast<void *>(plan_counts));
            return false;
        }

        if (!owned_moe_kernel_)
            owned_moe_kernel_ = KernelFactory::createMoEKernel(params_.device_id);
        if (!owned_moe_kernel_)
        {
            LOG_ERROR("[MoEDeviceRebalanceStage] Could not create the backend MoE kernel for request reset"
                      << " device=" << params_.device_id.to_string());
            return false;
        }

        const MoEKernelLaunchContext reset_launch{
            .stream = stream,
            .workspace = bound_workspace_,
        };
        if (!owned_moe_kernel_->resetDeviceRebalanceGraphTransactionForRequest(
                reset_launch,
                controller_state,
                command_headers,
                wave_states,
                plan_counts,
                static_cast<uint32_t>(commandBufferCount()),
                params_.config))
        {
            LOG_ERROR("[MoEDeviceRebalanceStage] Backend rejected the request-transaction reset"
                      << " device=" << params_.device_id.to_string()
                      << " stage=" << suffixFor(params_.stage_name));
            return false;
        }
        return true;
    }

    bool MoEDeviceRebalanceStage::runsPlanning() const
    {
        if (params_.phase == DeviceMoERebalanceStagePhase::JoinTransfer)
            return false;
        if (params_.phase == DeviceMoERebalanceStagePhase::PackCollectivePayloadAfterSideband ||
            params_.phase == DeviceMoERebalanceStagePhase::UnpackCollectivePayloadAfterSideband)
        {
            return false;
        }
        return collectsState() || runsController();
    }

    bool MoEDeviceRebalanceStage::runsApply() const
    {
        return params_.phase == DeviceMoERebalanceStagePhase::PlanCopyApply ||
               params_.phase == DeviceMoERebalanceStagePhase::Apply;
    }

    DeviceMoERebalanceStatusPublicationContract
    MoEDeviceRebalanceStage::statusPublicationContract() const
    {
        DeviceMoERebalanceStatusPublicationContract contract;
        const bool publishes_transfer_status = usesTransferSlotApply();
        const bool publishes_ready_apply_status = usesReadyWaveApply();

        /*
         * Keep this switch exhaustive. Adding a new phase must make its status
         * ownership explicit here before diagnostics are allowed to interpret
         * shared workspace. This prevents a newly allocated buffer from being
         * mistaken for a record produced by an unrelated graph transaction.
         */
        switch (params_.phase)
        {
            case DeviceMoERebalanceStagePhase::PlanCopyApply:
                contract.copy_status = publishes_transfer_status;
                contract.apply_status = publishes_ready_apply_status;
                break;
            case DeviceMoERebalanceStagePhase::PlanAndCopy:
                /*
                 * The current inline transport helper both publishes transfer
                 * completion and polls the newly ready wave on its transfer
                 * stream. A later split Apply stage may poll another buffered
                 * wave, but that does not erase this phase's own publication.
                 */
                contract.copy_status = publishes_transfer_status;
                contract.apply_status = publishes_transfer_status;
                break;
            case DeviceMoERebalanceStagePhase::PackCollectivePayloadAfterSideband:
                contract.copy_status = publishes_transfer_status;
                break;
            case DeviceMoERebalanceStagePhase::UnpackCollectivePayloadAfterSideband:
                contract.copy_status = publishes_transfer_status;
                contract.apply_status =
                    publishes_transfer_status &&
                    publishes_ready_apply_status;
                break;
            case DeviceMoERebalanceStagePhase::Apply:
                contract.apply_status = publishes_ready_apply_status;
                break;
            case DeviceMoERebalanceStagePhase::CollectState:
            case DeviceMoERebalanceStagePhase::PlanAndCopyAfterSideband:
            case DeviceMoERebalanceStagePhase::JoinTransfer:
                break;
        }
        return contract;
    }

    std::string MoEDeviceRebalanceStage::workspaceSuffix() const
    {
        return suffixFor(params_.workspace_name.empty()
                             ? params_.stage_name
                             : params_.workspace_name);
    }

    DeviceMoERebalanceTransferState *MoEDeviceRebalanceStage::transferState() const
    {
        return params_.transfer_state.get();
    }

    bool MoEDeviceRebalanceStage::ensureAsyncTransferState()
    {
        if (!usesTransferSlotApply())
            return true;
        if (!params_.transfer_state)
        {
            LOG_ERROR("[MoEDeviceRebalanceStage] Transfer-slot rebalance requires shared transfer state"
                      << " stage=" << suffixFor(params_.stage_name)
                      << " phase=" << phaseName(params_.phase));
            return false;
        }
        return params_.transfer_state->ensure(params_.device_id, workspaceSuffix());
    }

    bool MoEDeviceRebalanceStage::validateCommon(const char *context) const
    {
        const char *label = context ? context : "MoEDeviceRebalanceStage";
        if (!params_.device_id.is_gpu())
        {
            LOG_ERROR("[" << label << "] GPU device required, got "
                          << params_.device_id.to_string());
            return false;
        }
        if (!validateDeviceMoERebalanceConfig(params_.config))
        {
            LOG_ERROR("[" << label << "] Invalid device rebalance config"
                          << " layers=" << params_.config.num_layers
                          << " experts=" << params_.config.num_experts
                          << " top_k=" << params_.config.top_k
                          << " participant_id=" << params_.config.participant_id
                          << " participant_count=" << params_.config.participant_count
                          << " root_participant=" << params_.config.root_participant
                          << " window=" << params_.config.window_size_tokens);
            return false;
        }
        if (!params_.tp_ctx)
        {
            LOG_ERROR("[" << label << "] Missing LocalTP context");
            return false;
        }
        if (!params_.moe_runtime_table)
        {
            LOG_ERROR("[" << label << "] Missing MoE runtime table");
            return false;
        }
        if (params_.tp_ctx->degree() != static_cast<int>(params_.config.participant_count))
        {
            LOG_ERROR("[" << label << "] LocalTP degree/config participant count mismatch"
                          << " degree=" << params_.tp_ctx->degree()
                          << " participant_count=" << params_.config.participant_count);
            return false;
        }
        if (params_.tp_device_idx < 0 ||
            params_.tp_device_idx >= params_.tp_ctx->degree() ||
            params_.tp_device_idx != static_cast<int>(params_.config.participant_id))
        {
            LOG_ERROR("[" << label << "] Invalid TP participant index"
                          << " tp_device_idx=" << params_.tp_device_idx
                          << " config_participant_id=" << params_.config.participant_id
                          << " degree=" << params_.tp_ctx->degree());
            return false;
        }
        if (params_.moe_runtime_table->layerCount() != static_cast<int>(params_.config.num_layers))
        {
            LOG_ERROR("[" << label << "] Runtime table layer/config mismatch"
                          << " runtime_layers=" << params_.moe_runtime_table->layerCount()
                          << " config_layers=" << params_.config.num_layers);
            return false;
        }
        auto *device_table = dynamic_cast<DeviceMoERuntimeTable *>(params_.moe_runtime_table);
        if (!device_table || !device_table->isMirroredToDevice())
        {
            LOG_ERROR("[" << label << "] Device-side rebalance requires a mirrored DeviceMoERuntimeTable");
            return false;
        }
        if (device_table->expertCount() != static_cast<int>(params_.config.num_experts) ||
            device_table->topK() != static_cast<int>(params_.config.top_k))
        {
            LOG_ERROR("[" << label << "] Runtime table shape/config mismatch"
                          << " table_experts=" << device_table->expertCount()
                          << " config_experts=" << params_.config.num_experts
                          << " table_top_k=" << device_table->topK()
                          << " config_top_k=" << params_.config.top_k);
            return false;
        }
        if (usesCollectivePayloadLane() && params_.collective_payload_slot_bytes == 0)
        {
            LOG_ERROR("[" << label << "] Transfer-slot apply with a collective payload lane requires non-zero payload slot bytes");
            return false;
        }
        if (usesCollectivePayloadLane() && payloadSlotCapacity() == 0)
        {
            LOG_ERROR("[" << label << "] Transfer-slot apply with a collective payload lane requires non-zero payload slot capacity");
            return false;
        }
        return true;
    }

    bool MoEDeviceRebalanceStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "MoEDeviceRebalanceStage"))
            return false;
        if (!validateCommon("MoEDeviceRebalanceStage"))
            return false;

        void *stream = gpuStream();
        if (!stream)
        {
            LOG_ERROR("[MoEDeviceRebalanceStage] Explicit non-null GPU stream is required");
            return false;
        }

        if (params_.phase == DeviceMoERebalanceStagePhase::JoinTransfer)
        {
            if (!ensureAsyncTransferState())
                return false;
            auto *transfer_state = transferState();
            if (!transfer_state || !transfer_state->transferDoneEvent())
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] JoinTransfer requires transfer completion event");
                return false;
            }

            IWorkerGPUContext *gpu_ctx = nullptr;
            try
            {
                gpu_ctx = &GPUDeviceContextPool::instance().getContext(params_.device_id);
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Could not resolve worker GPU context for transfer join on "
                          << params_.device_id.to_string() << ": " << e.what());
                return false;
            }

            if (!gpu_ctx ||
                !gpu_ctx->waitEventChecked(transfer_state->transferDoneEvent(), stream))
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Failed to queue late transfer-stream join");
                return false;
            }

            return true;
        }

        if (!bound_workspace_)
        {
            LOG_ERROR("[MoEDeviceRebalanceStage] Workspace was not bound");
            return false;
        }

        auto *plan_entries = static_cast<DeviceMoERebalancePlanEntry *>(
            bound_workspace_->getBuffer(transferPlanBufferName()));
        auto *plan_count = static_cast<uint32_t *>(
            bound_workspace_->getBuffer(transferPlanCountBufferName()));
        auto *command_header = static_cast<DeviceMoERebalanceCommandBufferHeader *>(
            bound_workspace_->getBuffer(commandHeaderBufferName()));
        auto *controller_state = static_cast<DeviceMoERebalanceGraphControllerState *>(
            bound_workspace_->getBuffer(controllerStateBufferName()));
        auto *wave_state = static_cast<DeviceMoERebalanceWaveState *>(
            bound_workspace_->getBuffer(waveStateBufferName()));
        if (!plan_entries || !plan_count || !command_header || !controller_state || !wave_state)
        {
            LOG_ERROR("[MoEDeviceRebalanceStage] Missing workspace buffers"
                      << " plan_entries=" << static_cast<void *>(plan_entries)
                      << " plan_count=" << static_cast<void *>(plan_count)
                      << " command_header=" << static_cast<void *>(command_header)
                      << " controller_state=" << static_cast<void *>(controller_state)
                      << " wave_state=" << static_cast<void *>(wave_state)
                      << " phase=" << phaseName(params_.phase));
            return false;
        }

        uint64_t *local = nullptr;
        uint64_t *gathered = nullptr;
        DeviceMoERebalanceStatus *status = nullptr;
        const bool needs_status_workspace = runsPlanning();
        if (needs_status_workspace)
        {
            status = static_cast<DeviceMoERebalanceStatus *>(
                bound_workspace_->getBuffer(statusBufferName()));
            if (!status)
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Missing status workspace buffer"
                          << " status=" << static_cast<void *>(status)
                          << " phase=" << phaseName(params_.phase));
                return false;
            }
            if (runsPlanning())
            {
                local = static_cast<uint64_t *>(
                    bound_workspace_->getBuffer(localHistogramBufferName()));
                gathered = static_cast<uint64_t *>(
                    bound_workspace_->getBuffer(gatheredHistogramBufferName()));
                if (!local || !gathered)
                {
                    LOG_ERROR("[MoEDeviceRebalanceStage] Missing histogram workspace buffers"
                              << " local=" << static_cast<void *>(local)
                              << " gathered=" << static_cast<void *>(gathered)
                              << " phase=" << phaseName(params_.phase));
                    return false;
                }
            }
        }

        DeviceMoEExpertDirectoryEntry *local_directory = nullptr;
        DeviceMoERebalanceApplyStatus *copy_status = nullptr;
        DeviceMoERebalanceApplyStatus *gathered_copy_status = nullptr;
        DeviceMoERebalanceApplyStatus *apply_status = nullptr;
        DeviceMoERebalancePlanEntry *gathered_plan_entries = nullptr;
        DeviceMoERebalanceCommandBufferHeader *gathered_command_headers = nullptr;
        DeviceMoERebalanceWaveState *gathered_wave_states = nullptr;
        DeviceMoEExpertDirectoryEntry *local_source_descriptors = nullptr;
        uint8_t *local_transfer_payload = nullptr;
        uint8_t *gathered_transfer_payload = nullptr;
        if (usesReadyWaveApply())
        {
            apply_status = static_cast<DeviceMoERebalanceApplyStatus *>(
                bound_workspace_->getBuffer(applyStatusBufferName()));
            if (!apply_status)
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Missing ready-wave apply status workspace buffer"
                          << " apply_status=" << static_cast<void *>(apply_status));
                return false;
            }
        }
        if (usesTransferSlotApply())
        {
            copy_status = static_cast<DeviceMoERebalanceApplyStatus *>(
                bound_workspace_->getBuffer(copyStatusBufferName()));
            gathered_copy_status = static_cast<DeviceMoERebalanceApplyStatus *>(
                bound_workspace_->getBuffer(gatheredCopyStatusBufferName()));
            gathered_plan_entries = static_cast<DeviceMoERebalancePlanEntry *>(
                bound_workspace_->getBuffer(gatheredTransferPlanBufferName()));
            gathered_command_headers = static_cast<DeviceMoERebalanceCommandBufferHeader *>(
                bound_workspace_->getBuffer(gatheredCommandHeaderBufferName()));
            gathered_wave_states = static_cast<DeviceMoERebalanceWaveState *>(
                bound_workspace_->getBuffer(gatheredWaveStateBufferName()));
            if (!copy_status ||
                !gathered_copy_status ||
                !gathered_plan_entries ||
                !gathered_command_headers ||
                !gathered_wave_states)
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Missing transfer-slot metadata workspace buffers"
                          << " copy_status=" << static_cast<void *>(copy_status)
                          << " gathered_copy_status=" << static_cast<void *>(gathered_copy_status)
                          << " gathered_plan_entries=" << static_cast<void *>(gathered_plan_entries)
                          << " gathered_command_headers=" << static_cast<void *>(gathered_command_headers)
                          << " gathered_wave_states=" << static_cast<void *>(gathered_wave_states));
                return false;
            }
            if (usesCompactTransferSlots())
            {
                local_source_descriptors = static_cast<DeviceMoEExpertDirectoryEntry *>(
                    bound_workspace_->getBuffer(localSourceDescriptorsBufferName()));
                if (!local_source_descriptors)
                {
                    LOG_ERROR("[MoEDeviceRebalanceStage] Missing compact source-descriptor workspace buffers"
                              << " local_source_descriptors=" << static_cast<void *>(local_source_descriptors));
                    return false;
                }
            }
            if (usesCollectivePayloadLane())
            {
                local_transfer_payload = static_cast<uint8_t *>(
                    bound_workspace_->getBuffer(localTransferPayloadBufferName()));
                gathered_transfer_payload = static_cast<uint8_t *>(
                    bound_workspace_->getBuffer(gatheredTransferPayloadBufferName()));
                if (!local_transfer_payload || !gathered_transfer_payload)
                {
                    LOG_ERROR("[MoEDeviceRebalanceStage] Missing collective payload transfer workspace buffers"
                              << " local_transfer_payload=" << static_cast<void *>(local_transfer_payload)
                              << " gathered_transfer_payload=" << static_cast<void *>(gathered_transfer_payload));
                    return false;
                }
            }
            if (usesFixedPayloadTransfer())
            {
                local_directory = static_cast<DeviceMoEExpertDirectoryEntry *>(
                    bound_workspace_->getBuffer(localDirectoryBufferName()));
                if (!local_directory)
                {
                    LOG_ERROR("[MoEDeviceRebalanceStage] Missing fixed-payload local directory workspace buffer"
                              << " local_directory=" << static_cast<void *>(local_directory));
                    return false;
                }
            }
        }

        if (!owned_moe_kernel_)
            owned_moe_kernel_ = KernelFactory::createMoEKernel(params_.device_id);
        IMoEKernel *moe_kernel = owned_moe_kernel_.get();
        if (!moe_kernel)
        {
            LOG_ERROR("[MoEDeviceRebalanceStage] Failed to get MoE kernel for "
                      << params_.device_id.to_string());
            return false;
        }
        const MoEKernelLaunchContext compute_launch{
            .stream = stream,
            .workspace = bound_workspace_,
        };

        if (!moe_kernel->initializeDeviceRebalanceGraphController(
                compute_launch,
                controller_state,
                params_.config))
        {
            LOG_ERROR("[MoEDeviceRebalanceStage] Failed to initialize device rebalance graph controller state");
            return false;
        }

        if (PerfStatsCollector::isEnabled())
        {
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "device_rebalance_stage_mode",
                1.0,
                "decode",
                params_.device_id.to_string(),
                {{"stage", suffixFor(params_.stage_name)},
                 {"phase", phaseName(params_.phase)},
                 {"collects_state", boolString(collectsState())},
                 {"gathers_state_inline", boolString(gathersStateInline())},
                 {"uses_sideband_state",
                  boolString(
                      params_.phase ==
                          DeviceMoERebalanceStagePhase::PlanAndCopyAfterSideband)},
                 {"uses_transfer_slots", boolString(usesTransferSlotApply())},
                 {"command_buffer_count", std::to_string(commandBufferCount())}});
            if (runsPlanning())
            {
                const size_t local_histogram_entries = localHistogramEntries();
                const size_t gathered_histogram_entries = gatheredHistogramEntries();
                const size_t local_histogram_bytes =
                    local_histogram_entries * sizeof(uint64_t);
                const size_t gathered_histogram_bytes =
                    gathered_histogram_entries * sizeof(uint64_t);
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "device_rebalance_histogram_payload_bytes",
                    static_cast<double>(gathered_histogram_bytes),
                    "decode",
                    params_.device_id.to_string(),
                    {{"stage", suffixFor(params_.stage_name)},
                     {"phase", phaseName(params_.phase)},
                     {"histogram_layer_count", std::to_string(histogramLayerCount())},
                     {"num_experts", std::to_string(params_.config.num_experts)},
                     {"participant_count", std::to_string(params_.config.participant_count)},
                     {"layer_window_count", std::to_string(params_.config.layer_window_count)},
                     {"layer_wave_count", std::to_string(params_.config.layer_wave_count)},
                     {"local_entries", std::to_string(local_histogram_entries)},
                     {"gathered_entries", std::to_string(gathered_histogram_entries)},
                     {"local_bytes", std::to_string(local_histogram_bytes)},
                     {"gathered_bytes", std::to_string(gathered_histogram_bytes)}});
            }
        }

        DeviceMoELayerRuntime *runtime_layers =
            params_.moe_runtime_table->deviceLayerState(0);
        if (!runtime_layers)
        {
            LOG_ERROR("[MoEDeviceRebalanceStage] Runtime table returned null device state");
            return false;
        }

        IWorkerGPUContext *gpu_ctx = nullptr;
        DeviceMoERebalanceTransferState *transfer_state = nullptr;
        void *transfer_stream = nullptr;
        MoEKernelLaunchContext transfer_launch{};
        if (usesTransferSlotApply())
        {
            if (!ensureAsyncTransferState())
                return false;
            transfer_state = transferState();
            if (!transfer_state ||
                !transfer_state->transferStream() ||
                !transfer_state->computeReadyEvent() ||
                !transfer_state->transferDoneEvent())
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Transfer-slot rebalance requires an explicit transfer stream and events"
                          << " phase=" << phaseName(params_.phase));
                return false;
            }

            transfer_stream = transfer_state->transferStream();
            transfer_launch = MoEKernelLaunchContext{
                .stream = transfer_stream,
                .workspace = bound_workspace_,
            };

            try
            {
                gpu_ctx = &GPUDeviceContextPool::instance().getContext(params_.device_id);
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Could not resolve worker GPU context for transfer stream ordering on "
                          << params_.device_id.to_string() << ": " << e.what());
                return false;
            }

            if (PerfStatsCollector::isEnabled())
            {
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "device_rebalance_transfer_stream_path",
                    1.0,
                    "decode",
                    params_.device_id.to_string(),
                    {{"stage", suffixFor(params_.stage_name)},
                     {"phase", phaseName(params_.phase)},
                     {"path", "auxiliary_stream"},
                     {"graph_capture_active", boolString(isGraphCaptureActive())}});
            }
        }

        auto project_domain_commands =
            [&](const MoEKernelLaunchContext &launch) -> bool
        {
            if (!usesTransferSlotApply())
                return true;
            if (!gathered_plan_entries || !gathered_command_headers || !gathered_wave_states)
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Domain command projection requires gathered command and wave buffers");
                return false;
            }
            if (!moe_kernel->projectDeviceRebalanceDomainCommands(
                    launch,
                    gathered_plan_entries,
                    gathered_command_headers,
                    static_cast<uint32_t>(transferPlanCapacity()),
                    plan_entries,
                    command_header,
                    params_.config,
                    status,
                    static_cast<uint32_t>(payloadSlotCapacity()),
                    static_cast<uint32_t>(commandBufferCount()),
                    gathered_wave_states,
                    wave_state,
                    runtime_layers,
                    params_.local_transfer_slots,
                    params_.local_transfer_slot_count))
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Failed to project gathered root commands into local apply ABI");
                return false;
            }
            if (PerfStatsCollector::isEnabled())
            {
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "device_rebalance_domain_command_projection",
                    1.0,
                    "decode",
                    params_.device_id.to_string(),
                    {{"stage", suffixFor(params_.stage_name)},
                     {"phase", phaseName(params_.phase)},
                     {"root_participant", std::to_string(params_.config.root_participant)},
                     {"participant", std::to_string(params_.config.participant_id)},
                     {"command_buffer_count", std::to_string(commandBufferCount())}});
            }
            return true;
        };

        auto join_transfer_stream_to_capture_stream = [&](const char *context) -> bool
        {
            if (!usesTransferSlotApply())
                return true;
            if (!gpu_ctx || !transfer_state)
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] " << context
                          << " requires transfer stream state");
                return false;
            }
            if (!gpu_ctx->recordEventChecked(transfer_state->transferDoneEvent(),
                                             transfer_stream))
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Failed to record "
                          << context << " transfer-stream completion event");
                return false;
            }
            /*
             * `join_transfer_stream_after_copy` means the public stage stream is
             * the completion contract for this stage.  Captured graphs need the
             * event wait recorded into the graph, while stream-only maintenance
             * replays need the same wait enqueued immediately so the
             * orchestrator's completion event cannot race auxiliary-stream
             * RCCL/P2P payload work.
             */
            if (params_.join_transfer_stream_after_copy &&
                !gpu_ctx->waitEventChecked(transfer_state->transferDoneEvent(), stream))
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Failed to join "
                          << context << " transfer stream back to the stage stream");
                return false;
            }
            return true;
        };

        auto has_incoming_payload_edges = [&]() -> bool
        {
            return usesCollectivePayloadLane();
        };

        auto apply_published_transfer_wave =
            [&](const MoEKernelLaunchContext &launch,
                const char *context) -> bool
        {
            if (!usesReadyWaveApply())
                return true;
            if (!apply_status)
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] " << context
                          << " requires ready-wave apply status");
                return false;
            }
            if (!moe_kernel->applyReadyDeviceRebalanceWave(
                    launch,
                    runtime_layers,
                    plan_entries,
                    plan_count,
                    static_cast<uint32_t>(transferPlanCapacity()),
                    params_.local_transfer_slots,
                    params_.local_transfer_slot_count,
                    params_.config,
                    apply_status,
                    controller_state,
                    command_header,
                    /*target_layer=*/-1,
                    static_cast<uint32_t>(commandBufferCount())))
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Failed to apply ready device rebalance wave after "
                          << context);
                return false;
            }
            return true;
        };

        auto gather_copy_status = [&](const char *context) -> bool
        {
            if (!copy_status || !gathered_copy_status)
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] " << context
                          << " requires copy-status workspace buffers");
                return false;
            }
            static_assert((sizeof(DeviceMoERebalanceApplyStatus) % sizeof(int32_t)) == 0);
            const size_t copy_status_int32_words =
                sizeof(DeviceMoERebalanceApplyStatus) / sizeof(int32_t);
            if (!params_.tp_ctx->allgatherRawOnStream(
                    copy_status,
                    gathered_copy_status,
                    copy_status_int32_words,
                    CollectiveDataType::INT32,
                    params_.tp_device_idx,
                    transfer_stream,
                    workspaceSuffix() + "_copy_status"))
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] " << context
                          << " copy-status allgather failed");
                return false;
            }
            return true;
        };

        /**
         * Snapshot the source side of the current projected command buffer.
         *
         * This helper is deliberately separate from payload transport. The
         * runtime expert directory is mutable: ordinary decode can evict and
         * replace a cache slot after this graph transaction completes.
         * Therefore source selection and source dereference remain adjacent
         * inside one replay, turning every selected source into immutable bytes
         * before payload transport begins.
         */
        auto prepare_local_payload = [&]() -> bool
        {
            if (!usesCollectivePayloadLane())
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Payload preparation requires a collective payload lane");
                return false;
            }

            if (usesCompactTransferSlots())
            {
                if (!moe_kernel->packDeviceRebalanceSourceDescriptors(
                        transfer_launch,
                        runtime_layers,
                        plan_entries,
                        command_header,
                        static_cast<uint32_t>(transferPlanCapacity()),
                        local_source_descriptors,
                        params_.config,
                        controller_state,
                        static_cast<uint32_t>(commandBufferCount())))
                {
                    LOG_ERROR("[MoEDeviceRebalanceStage] Compact rebalance source-descriptor pack failed");
                    return false;
                }

                if (!moe_kernel->packDeviceRebalanceCompactPayloads(
                        transfer_launch,
                        plan_entries,
                        command_header,
                        static_cast<uint32_t>(transferPlanCapacity()),
                        local_source_descriptors,
                        local_transfer_payload,
                        static_cast<uint32_t>(collectivePayloadSlotCount()),
                        params_.collective_payload_slot_bytes,
                        params_.config,
                        copy_status,
                        controller_state,
                        static_cast<uint32_t>(commandBufferCount())))
                {
                    LOG_ERROR("[MoEDeviceRebalanceStage] Compact rebalance payload pack failed");
                    return false;
                }
            }
            else
            {
                if (!moe_kernel->packDeviceRebalanceCollectivePayloads(
                        transfer_launch,
                        gathered_plan_entries,
                        gathered_command_headers,
                        static_cast<uint32_t>(transferPlanCapacity()),
                        local_directory,
                        local_transfer_payload,
                        static_cast<uint32_t>(collectivePayloadSlotCount()),
                        params_.collective_payload_slot_bytes,
                        params_.config,
                        copy_status,
                        controller_state,
                        static_cast<uint32_t>(commandBufferCount())))
                {
                    LOG_ERROR("[MoEDeviceRebalanceStage] Collective rebalance payload pack failed");
                    return false;
                }
            }
            return true;
        };

        /**
         * Transport and apply a command buffer whose source payload is stable.
         *
         * @param transfer_stream_already_ordered
         *     True when command metadata projection has already established the
         *     compute-to-transfer event edge for this invocation.
         *
         * There is intentionally no "prepare if needed" parameter. Callers
         * that own a prepare-and-transfer transaction must invoke
         * `prepare_local_payload()` explicitly first. As a result this helper,
         * and therefore the transport portion of the transaction, has no
         * executable branch that can dereference the live runtime directory.
         */
        auto transfer_prepared_payload =
            [&](bool transfer_stream_already_ordered) -> bool
        {
            if (!usesCollectivePayloadLane())
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Prepared payload transfer requires a collective payload lane");
                return false;
            }
            if (!gpu_ctx || !transfer_state)
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Prepared payload transfer requires transfer stream state");
                return false;
            }

            if (!transfer_stream_already_ordered &&
                (!gpu_ctx->recordEventChecked(transfer_state->computeReadyEvent(), stream) ||
                 !gpu_ctx->waitEventChecked(transfer_state->computeReadyEvent(), transfer_stream)))
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Failed to queue prepared payload compute-to-transfer dependency");
                return false;
            }

            if (!params_.tp_ctx->allgatherRawOnStream(
                    local_transfer_payload,
                    gathered_transfer_payload,
                    collectivePayloadLocalBytes(),
                    CollectiveDataType::INT8,
                    params_.tp_device_idx,
                    transfer_stream,
                    workspaceSuffix() + "_transfer_payload"))
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Transfer-payload allgather failed");
                return false;
            }

            if (has_incoming_payload_edges() &&
                !moe_kernel->unpackDeviceRebalanceCollectivePayloads(
                    transfer_launch,
                    plan_entries,
                    plan_count,
                    static_cast<uint32_t>(transferPlanCapacity()),
                    command_header,
                    gathered_transfer_payload,
                    static_cast<uint32_t>(collectivePayloadSlotCount()),
                    params_.collective_payload_slot_bytes,
                    params_.local_transfer_slots,
                    params_.local_transfer_slot_count,
                    params_.config,
                    copy_status,
                    controller_state,
                    static_cast<uint32_t>(commandBufferCount())))
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Collective rebalance payload unpack failed");
                return false;
            }

            if (!gather_copy_status("prepared payload copy"))
                return false;

            if (!moe_kernel->publishDeviceRebalanceTransferComplete(
                    transfer_launch,
                    controller_state,
                    command_header,
                    wave_state,
                    copy_status,
                    plan_entries,
                    static_cast<uint32_t>(transferPlanCapacity()),
                    gathered_copy_status,
                    params_.config,
                    static_cast<uint32_t>(commandBufferCount())))
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Failed to publish device rebalance transfer completion");
                return false;
            }

            if (!apply_published_transfer_wave(
                    transfer_launch,
                    "prepared payload transfer"))
                return false;

            return join_transfer_stream_to_capture_stream("prepared payload transfer");
        };

        if (params_.phase == DeviceMoERebalanceStagePhase::PackCollectivePayloadAfterSideband)
        {
            if (!usesFixedPayloadTransfer())
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] PackCollectivePayloadAfterSideband requires fixed payload transfer slots");
                return false;
            }
            if (!project_domain_commands(compute_launch))
                return false;

            if (!moe_kernel->packDeviceRebalanceCollectivePayloads(
                    compute_launch,
                    gathered_plan_entries,
                    gathered_command_headers,
                    static_cast<uint32_t>(transferPlanCapacity()),
                    local_directory,
                    local_transfer_payload,
                    static_cast<uint32_t>(collectivePayloadSlotCount()),
                    params_.collective_payload_slot_bytes,
                    params_.config,
                    copy_status,
                    controller_state,
                    static_cast<uint32_t>(commandBufferCount())))
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Sideband collective rebalance payload pack failed");
                return false;
            }
            return true;
        }

        if (params_.phase == DeviceMoERebalanceStagePhase::UnpackCollectivePayloadAfterSideband)
        {
            if (!usesFixedPayloadTransfer())
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] UnpackCollectivePayloadAfterSideband requires fixed payload transfer slots");
                return false;
            }
            if (!gpu_ctx || !transfer_state)
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Sideband collective payload unpack requires transfer stream state");
                return false;
            }

            if ((!gpu_ctx->recordEventChecked(transfer_state->computeReadyEvent(), stream) ||
                 !gpu_ctx->waitEventChecked(transfer_state->computeReadyEvent(), transfer_stream)))
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Failed to queue sideband payload compute-to-transfer dependency");
                return false;
            }

            if (has_incoming_payload_edges() &&
                !moe_kernel->unpackDeviceRebalanceCollectivePayloads(
                    transfer_launch,
                    plan_entries,
                    plan_count,
                    static_cast<uint32_t>(transferPlanCapacity()),
                    command_header,
                    gathered_transfer_payload,
                    static_cast<uint32_t>(collectivePayloadSlotCount()),
                    params_.collective_payload_slot_bytes,
                    params_.local_transfer_slots,
                    params_.local_transfer_slot_count,
                    params_.config,
                    copy_status,
                    controller_state,
                    static_cast<uint32_t>(commandBufferCount())))
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Sideband collective rebalance payload unpack failed");
                return false;
            }

            if (!gather_copy_status("sideband collective payload unpack"))
                return false;

            if (!moe_kernel->publishDeviceRebalanceTransferComplete(
                    transfer_launch,
                    controller_state,
                    command_header,
                    wave_state,
                    copy_status,
                    plan_entries,
                    static_cast<uint32_t>(transferPlanCapacity()),
                    gathered_copy_status,
                    params_.config,
                    static_cast<uint32_t>(commandBufferCount())))
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Failed to publish sideband collective rebalance transfer completion");
                return false;
            }

            if (!apply_published_transfer_wave(
                    transfer_launch,
                    "sideband collective payload unpack"))
                return false;

            if (!gpu_ctx->recordEventChecked(transfer_state->transferDoneEvent(),
                                             transfer_stream))
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Failed to record sideband transfer completion event");
                return false;
            }
            if (params_.join_transfer_stream_after_copy &&
                !gpu_ctx->waitEventChecked(transfer_state->transferDoneEvent(), stream))
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Failed to join sideband transfer stream back to the stage stream");
                return false;
            }
            return true;
        }

        if (runsPlanning())
        {
            if (collectsState())
            {
                if (!moe_kernel->packDeviceRebalanceHistograms(
                        compute_launch,
                        runtime_layers,
                        local,
                        params_.config,
                        wave_state,
                        controller_state,
                        static_cast<uint32_t>(commandBufferCount())))
                {
                    LOG_ERROR("[MoEDeviceRebalanceStage] Failed to pack local runtime histograms");
                    return false;
                }

                if (usesFixedPayloadTransfer())
                {
                    if (!moe_kernel->packDeviceRebalanceDirectory(
                            compute_launch,
                            runtime_layers,
                            local_directory,
                            params_.config))
                    {
                        LOG_ERROR("[MoEDeviceRebalanceStage] Failed to pack local expert directory");
                        return false;
                    }
                }
            }

            if (gathersStateInline())
            {
                static_assert(sizeof(uint64_t) == 2 * sizeof(int32_t));
                const size_t int32_words = localHistogramEntries() * 2u;
                if (!params_.tp_ctx->allgatherRawOnStream(
                        local,
                        gathered,
                        int32_words,
                        CollectiveDataType::INT32,
                        params_.tp_device_idx,
                        stream,
                        workspaceSuffix() + "_histogram"))
                {
                    LOG_ERROR("[MoEDeviceRebalanceStage] Histogram allgather failed");
                    return false;
                }

            }

            if (runsController())
            {
                if (!moe_kernel->runDeviceRebalanceController(
                        compute_launch,
                        runtime_layers,
                        gathered,
                        status,
                        params_.config,
                        plan_entries,
                        plan_count,
                        static_cast<uint32_t>(transferPlanCapacity()),
                        static_cast<uint32_t>(payloadSlotCapacity()),
                        command_header,
                        wave_state,
                        controller_state,
                        static_cast<uint32_t>(commandBufferCount()),
                        params_.local_transfer_slots,
                        params_.local_transfer_slot_count))
                {
                    LOG_ERROR("[MoEDeviceRebalanceStage] Device rebalance controller failed");
                    return false;
                }

                if (usesTransferSlotApply())
                {
                    if ((!gpu_ctx->recordEventChecked(transfer_state->computeReadyEvent(), stream) ||
                         !gpu_ctx->waitEventChecked(transfer_state->computeReadyEvent(), transfer_stream)))
                    {
                        LOG_ERROR("[MoEDeviceRebalanceStage] Failed to queue compute-to-transfer stream dependency");
                        return false;
                    }
                    {
                        static_assert((sizeof(DeviceMoERebalancePlanEntry) % sizeof(int32_t)) == 0);
                        static_assert((sizeof(DeviceMoERebalanceCommandBufferHeader) % sizeof(int32_t)) == 0);
                        static_assert((sizeof(DeviceMoERebalanceWaveState) % sizeof(int32_t)) == 0);
                        const size_t plan_int32_words =
                            (commandBufferCount() * transferPlanCapacity() * sizeof(DeviceMoERebalancePlanEntry)) /
                            sizeof(int32_t);
                        const size_t header_int32_words =
                            (commandBufferCount() * sizeof(DeviceMoERebalanceCommandBufferHeader)) / sizeof(int32_t);
                        const size_t wave_int32_words =
                            (commandBufferCount() * sizeof(DeviceMoERebalanceWaveState)) / sizeof(int32_t);
                        if (!params_.tp_ctx->allgatherRawOnStream(
                                plan_entries,
                                gathered_plan_entries,
                                plan_int32_words,
                                CollectiveDataType::INT32,
                                params_.tp_device_idx,
                                transfer_stream,
                                workspaceSuffix() + "_transfer_plan"))
                        {
                            LOG_ERROR("[MoEDeviceRebalanceStage] Transfer-plan allgather failed");
                            return false;
                        }
                        if (!params_.tp_ctx->allgatherRawOnStream(
                                command_header,
                                gathered_command_headers,
                                header_int32_words,
                                CollectiveDataType::INT32,
                                params_.tp_device_idx,
                                transfer_stream,
                                workspaceSuffix() + "_transfer_header"))
                        {
                            LOG_ERROR("[MoEDeviceRebalanceStage] Transfer-header allgather failed");
                            return false;
                        }
                        if (!params_.tp_ctx->allgatherRawOnStream(
                                wave_state,
                                gathered_wave_states,
                                wave_int32_words,
                                CollectiveDataType::INT32,
                                params_.tp_device_idx,
                                transfer_stream,
                                workspaceSuffix() + "_transfer_wave"))
                        {
                            LOG_ERROR("[MoEDeviceRebalanceStage] Transfer-wave allgather failed");
                            return false;
                        }

                        if (!project_domain_commands(transfer_launch))
                            return false;

                        if (params_.phase == DeviceMoERebalanceStagePhase::PlanAndCopyAfterSideband)
                        {
                            if (!join_transfer_stream_to_capture_stream("planned command metadata"))
                                return false;
                            return true;
                        }

                        /*
                         * Inline producer phases prepare and transport
                         * atomically. Keeping the calls adjacent preserves
                         * immutable source ownership without giving the
                         * transport helper a route back to runtime state.
                         */
                        if (!prepare_local_payload())
                            return false;
                        if (!transfer_prepared_payload(
                                /*transfer_stream_already_ordered=*/true))
                            return false;
                    }
                }
            }
        }

        /*
         * An inline transfer transaction publishes and applies its ready wave
         * inside `transfer_prepared_payload()` on the transfer stream.  Polling
         * again here would be a second state-machine transition in the same
         * graph replay: the first poll consumes the ready wave, while the
         * redundant poll observes no work and can overwrite the terminal apply
         * record.  Non-transfer deferred-apply configurations still need this
         * compute-stream apply step because they have no transport helper.
         */
        const bool inline_transfer_already_applied =
            usesTransferSlotApply() && runsController();
        if (runsApply() &&
            usesReadyWaveApply() &&
            !inline_transfer_already_applied)
        {
            if (!moe_kernel->applyReadyDeviceRebalanceWave(
                    compute_launch,
                    runtime_layers,
                    plan_entries,
                    plan_count,
                    static_cast<uint32_t>(transferPlanCapacity()),
                    params_.local_transfer_slots,
                    params_.local_transfer_slot_count,
                    params_.config,
                    apply_status,
                    controller_state,
                    command_header,
                    params_.apply_layer_idx,
                    static_cast<uint32_t>(commandBufferCount())))
            {
                LOG_ERROR("[MoEDeviceRebalanceStage] Device rebalance ready-wave apply failed");
                return false;
            }
        }

        LOG_DEBUG("[MoEDeviceRebalanceStage] Enqueued graph-side rebalance"
                  << " stage=" << suffixFor(params_.stage_name)
                  << " device=" << params_.device_id.to_string()
                  << " participant=" << params_.tp_device_idx
                  << " layers=" << params_.config.num_layers
                  << " experts=" << params_.config.num_experts
                  << " phase=" << phaseName(params_.phase));
        return true;
    }

    size_t MoEDeviceRebalanceStage::estimatedMemoryBytes() const
    {
        if (params_.phase == DeviceMoERebalanceStagePhase::JoinTransfer)
            return 0;
        size_t bytes =
            (localHistogramEntries() + gatheredHistogramEntries()) * sizeof(uint64_t) +
            commandBufferCount() * transferPlanCapacity() * sizeof(DeviceMoERebalancePlanEntry) +
            commandBufferCount() * sizeof(uint32_t) +
            commandBufferCount() * sizeof(DeviceMoERebalanceCommandBufferHeader) +
            sizeof(DeviceMoERebalanceGraphControllerState) +
            commandBufferCount() * sizeof(DeviceMoERebalanceWaveState) +
            sizeof(DeviceMoERebalanceStatus);
        if (usesReadyWaveApply())
        {
            bytes += sizeof(DeviceMoERebalanceApplyStatus);
        }
        if (usesTransferSlotApply())
        {
            bytes += sizeof(DeviceMoERebalanceApplyStatus);
            bytes += static_cast<size_t>(params_.config.participant_count) *
                     sizeof(DeviceMoERebalanceApplyStatus);
            bytes += transferPlanCapacity() *
                         static_cast<size_t>(params_.config.participant_count) *
                         commandBufferCount() *
                         sizeof(DeviceMoERebalancePlanEntry) +
                     static_cast<size_t>(params_.config.participant_count) *
                         commandBufferCount() *
                         sizeof(DeviceMoERebalanceCommandBufferHeader) +
                     static_cast<size_t>(params_.config.participant_count) *
                         commandBufferCount() *
                         sizeof(DeviceMoERebalanceWaveState);
            if (usesCompactTransferSlots())
            {
                bytes +=
                    localSourceDescriptorEntries() *
                    sizeof(DeviceMoEExpertDirectoryEntry);
            }
            if (usesCollectivePayloadLane())
            {
                bytes +=
                    collectivePayloadLocalBytes() +
                    collectivePayloadGatheredBytes();
            }
            if (usesFixedPayloadTransfer())
            {
                bytes +=
                    localDirectoryEntries() * sizeof(DeviceMoEExpertDirectoryEntry);
            }
        }
        return bytes;
    }

    bool MoEDeviceRebalanceStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
#ifdef HAVE_CUDA
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#ifdef HAVE_ROCM
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
    }

    bool MoEDeviceRebalanceStage::isCollectiveStage() const
    {
        /*
         * Inline state gathering always launches the histogram allgather.
         * Transfer-backed controller phases additionally gather command, wave,
         * payload, and completion records. Sideband pack/unpack, Apply,
         * JoinTransfer, and CollectState remain participant-local.
         */
        return gathersStateInline() ||
               (runsController() && usesTransferSlotApply());
    }

    bool MoEDeviceRebalanceStage::isGraphCapturable() const
    {
        if (!validateDeviceMoERebalanceConfig(params_.config))
            return false;
        if (!params_.device_id.is_gpu() || !params_.tp_ctx || !params_.moe_runtime_table)
            return false;
        if (params_.tp_ctx->degree() <= 1 ||
            params_.tp_ctx->degree() != static_cast<int>(params_.config.participant_count))
            return false;
        if (params_.tp_device_idx < 0 ||
            params_.tp_device_idx >= params_.tp_ctx->degree() ||
            params_.tp_device_idx != static_cast<int>(params_.config.participant_id))
            return false;
        auto *device_table = dynamic_cast<DeviceMoERuntimeTable *>(params_.moe_runtime_table);
        if (!device_table || !device_table->isMirroredToDevice())
            return false;

        bool backend_supported = false;
#ifdef HAVE_CUDA
        backend_supported = backend_supported || params_.device_id.is_cuda();
#endif
#ifdef HAVE_ROCM
        backend_supported = backend_supported || params_.device_id.is_rocm();
#endif
        return backend_supported &&
               params_.tp_ctx->supportsRawAllgatherOnStreamGraphCapture();
    }

    bool MoEDeviceRebalanceStage::prepareGraphLaunch(IDeviceContext *ctx, void *stream)
    {
        (void)ctx;
        if (!stream)
        {
            LOG_ERROR("[MoEDeviceRebalanceStage] Graph launch preparation requires an explicit non-null stream");
            return false;
        }
        setGPUStream(stream);
        if (!usesTransferSlotApply())
            return true;
        auto *transfer_state = transferState();
        if (!transfer_state)
        {
            LOG_ERROR("[MoEDeviceRebalanceStage] Pre-capture transfer-stream fence requires shared transfer state"
                      << " stage=" << suffixFor(params_.stage_name)
                      << " device=" << params_.device_id.to_string());
            return false;
        }
        if (!transfer_state->prepareForCapture(
                params_.device_id,
                workspaceSuffix(),
                stream))
        {
            LOG_ERROR("[MoEDeviceRebalanceStage] Pre-capture transfer-stream event fence failed"
                      << " stage=" << suffixFor(params_.stage_name)
                      << " phase=" << phaseName(params_.phase)
                      << " device=" << params_.device_id.to_string()
                      << " participant=" << params_.tp_device_idx);
            return false;
        }
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "device_rebalance_precapture_transfer_stream_event_fence",
            1.0,
            "decode",
            params_.device_id.to_string(),
            {{"stage", suffixFor(params_.stage_name)},
             {"phase", phaseName(params_.phase)},
             {"participant", std::to_string(params_.tp_device_idx)}});
        return true;
    }

    StageDumpInfo MoEDeviceRebalanceStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        info.addScalarInt("num_layers", static_cast<int>(params_.config.num_layers))
            .addScalarInt("num_experts", static_cast<int>(params_.config.num_experts))
            .addScalarInt("top_k", static_cast<int>(params_.config.top_k))
            .addScalarInt("participant_id", static_cast<int>(params_.config.participant_id))
            .addScalarInt("participant_count", static_cast<int>(params_.config.participant_count))
            .addScalarInt("root_participant", static_cast<int>(params_.config.root_participant))
            .addScalarInt("window_size_tokens", static_cast<int>(params_.config.window_size_tokens))
            .addScalarInt("max_hot_replicas_per_participant",
                          static_cast<int>(params_.config.max_hot_replicas_per_participant))
            .addScalarInt("layer_window_start",
                          static_cast<int>(params_.config.layer_window_start))
            .addScalarInt("layer_window_count",
                          static_cast<int>(params_.config.layer_window_count))
            .addScalarInt("layer_wave_count",
                          static_cast<int>(params_.config.layer_wave_count))
            .addScalarInt("histogram_layer_count",
                          static_cast<int>(histogramLayerCount()))
            .addScalarInt("apply_layer_idx", params_.apply_layer_idx)
            .addScalarBool("transfer_slots",
                           usesTransferSlotApply())
            .addScalarInt("collective_payload_slot_bytes",
                          static_cast<int>(params_.collective_payload_slot_bytes))
            .addScalarInt("collective_payload_slot_capacity",
                          static_cast<int>(params_.collective_payload_slot_capacity))
            .addScalarInt("payload_slot_capacity",
                          static_cast<int>(payloadSlotCapacity()))
            .addScalarInt("collective_payload_slot_count",
                          static_cast<int>(collectivePayloadSlotCount()))
            .addScalarInt("transfer_plan_entries",
                          static_cast<int>(transferPlanEntries()))
            .addScalarInt("transfer_plan_capacity",
                          static_cast<int>(transferPlanCapacity()))
            .addScalarInt("command_buffer_count",
                          static_cast<int>(commandBufferCount()))
            .addScalarInt("command_header_bytes",
                          static_cast<int>(sizeof(DeviceMoERebalanceCommandBufferHeader)))
            .addScalarInt("controller_state_bytes",
                          static_cast<int>(sizeof(DeviceMoERebalanceGraphControllerState)))
            .addScalarInt("wave_state_bytes",
                          static_cast<int>(sizeof(DeviceMoERebalanceWaveState)))
            .addScalarInt("flags", static_cast<int>(params_.config.flags));
        return info;
    }

    WorkspaceRequirements MoEDeviceRebalanceStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        (void)m;
        (void)n;
        (void)k;
        WorkspaceRequirements reqs;
        if (!validateDeviceMoERebalanceConfig(params_.config))
            return reqs;
        if (params_.phase == DeviceMoERebalanceStagePhase::JoinTransfer)
            return reqs;

        reqs.buffers.push_back({localHistogramBufferName(),
                                localHistogramEntries() * sizeof(uint64_t),
                                256,
                                true});
        reqs.buffers.push_back({gatheredHistogramBufferName(),
                                gatheredHistogramEntries() * sizeof(uint64_t),
                                256,
                                true});
        reqs.buffers.push_back({transferPlanBufferName(),
                                commandBufferCount() * transferPlanCapacity() *
                                    sizeof(DeviceMoERebalancePlanEntry),
                                256,
                                true});
        reqs.buffers.push_back({transferPlanCountBufferName(),
                                commandBufferCount() * sizeof(uint32_t),
                                256,
                                true});
        reqs.buffers.push_back({commandHeaderBufferName(),
                                commandBufferCount() *
                                    sizeof(DeviceMoERebalanceCommandBufferHeader),
                                256,
                                true});
        reqs.buffers.push_back({controllerStateBufferName(),
                                sizeof(DeviceMoERebalanceGraphControllerState),
                                256,
                                true});
        reqs.buffers.push_back({waveStateBufferName(),
                                commandBufferCount() * sizeof(DeviceMoERebalanceWaveState),
                                256,
                                true});
        reqs.buffers.push_back({statusBufferName(),
                                sizeof(DeviceMoERebalanceStatus),
                                256,
                                true});
        if (usesReadyWaveApply())
        {
            reqs.buffers.push_back({applyStatusBufferName(),
                                    sizeof(DeviceMoERebalanceApplyStatus),
                                    256,
                                    true});
        }
        if (usesTransferSlotApply())
        {
            reqs.buffers.push_back({copyStatusBufferName(),
                                    sizeof(DeviceMoERebalanceApplyStatus),
                                    256,
                                    true});
            reqs.buffers.push_back({gatheredCopyStatusBufferName(),
                                    static_cast<size_t>(params_.config.participant_count) *
                                        sizeof(DeviceMoERebalanceApplyStatus),
                                    256,
                                    true});
            reqs.buffers.push_back({gatheredTransferPlanBufferName(),
                                    transferPlanCapacity() *
                                        commandBufferCount() *
                                        static_cast<size_t>(params_.config.participant_count) *
                                        sizeof(DeviceMoERebalancePlanEntry),
                                    256,
                                    true});
            reqs.buffers.push_back({gatheredCommandHeaderBufferName(),
                                    static_cast<size_t>(params_.config.participant_count) *
                                        commandBufferCount() *
                                        sizeof(DeviceMoERebalanceCommandBufferHeader),
                                    256,
                                    true});
            reqs.buffers.push_back({gatheredWaveStateBufferName(),
                                    static_cast<size_t>(params_.config.participant_count) *
                                        commandBufferCount() *
                                        sizeof(DeviceMoERebalanceWaveState),
                                    256,
                                    true});
            if (usesCompactTransferSlots())
            {
                reqs.buffers.push_back({localSourceDescriptorsBufferName(),
                                        localSourceDescriptorEntries() *
                                            sizeof(DeviceMoEExpertDirectoryEntry),
                                        256,
                                        true});
            }
            if (usesCollectivePayloadLane())
            {
                reqs.buffers.push_back({localTransferPayloadBufferName(),
                                        collectivePayloadLocalBytes(),
                                        256,
                                        true});
                reqs.buffers.push_back({gatheredTransferPayloadBufferName(),
                                        collectivePayloadGatheredBytes(),
                                        256,
                                        true});
            }
            if (usesFixedPayloadTransfer())
            {
                reqs.buffers.push_back({localDirectoryBufferName(),
                                        localDirectoryEntries() * sizeof(DeviceMoEExpertDirectoryEntry),
                                        256,
                                        true});
            }
        }
        return reqs;
    }

    void MoEDeviceRebalanceStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        bound_workspace_ = workspace;
    }

    void MoEDeviceRebalanceStage::unbindWorkspace()
    {
        bound_workspace_ = nullptr;
    }

} // namespace llaminar2
