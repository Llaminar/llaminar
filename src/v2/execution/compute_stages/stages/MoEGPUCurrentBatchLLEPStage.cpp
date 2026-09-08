/**
 * @file MoEGPUCurrentBatchLLEPStage.cpp
 * @brief GPU current-batch LLEP graph-phase implementation.
 *
 * The implementation intentionally contains no NCCL/RCCL call. PlanAndPack
 * publishes a persistent byte lane; the model graph attaches that lane to a
 * normal collective; UnpackApplyAndAssign consumes the gathered bytes. Keeping
 * this boundary explicit lets graph capture retain one production-shaped
 * sparse MoE transaction and makes collective count visible to PerfStats.
 */

#include "MoEGPUCurrentBatchLLEPStage.h"

#include "MoEDeviceRebalanceStage.h"
#include "../../../collective/ILocalTPContext.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../execution/moe/MoERuntimeTable.h"
#include "../../../kernels/IMoEKernel.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        using KernelFactory = llaminar::v2::kernels::KernelFactory;

        /** @return Stable diagnostic name for one graph phase. */
        const char *phaseName(GPUCurrentBatchLLEPPhase phase) noexcept
        {
            switch (phase)
            {
            case GPUCurrentBatchLLEPPhase::PlanAndPack:
                return "plan_and_pack";
            case GPUCurrentBatchLLEPPhase::UnpackApplyAndAssign:
                return "unpack_apply_and_assign";
            }
            return "unknown";
        }

        /** @return @p value narrowed after saturating at UINT32_MAX. */
        std::uint32_t boundedU32(std::size_t value) noexcept
        {
            return static_cast<std::uint32_t>(
                std::min<std::size_t>(
                    value,
                    static_cast<std::size_t>(
                        std::numeric_limits<std::uint32_t>::max())));
        }
    } // namespace

    MoEGPUCurrentBatchLLEPStage::MoEGPUCurrentBatchLLEPStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if (params_.stage_name.empty())
        {
            params_.stage_name =
                params_.phase == GPUCurrentBatchLLEPPhase::PlanAndPack
                    ? "moe_current_batch_llep_plan_pack"
                    : "moe_current_batch_llep_unpack_apply_assign";
        }
    }

    MoEGPUCurrentBatchLLEPStage::~MoEGPUCurrentBatchLLEPStage() = default;

    std::string MoEGPUCurrentBatchLLEPStage::name() const
    {
        return params_.stage_name;
    }

    bool MoEGPUCurrentBatchLLEPStage::supportsBackend(
        ComputeBackendType backend) const
    {
        switch (backend)
        {
#if defined(HAVE_CUDA)
        case ComputeBackendType::GPU_CUDA:
            return params_.device_id.is_cuda();
#endif
#if defined(HAVE_ROCM)
        case ComputeBackendType::GPU_ROCM:
            return params_.device_id.is_rocm();
#endif
        default:
            return false;
        }
    }

    bool MoEGPUCurrentBatchLLEPStage::isGraphCapturable() const
    {
        return bound_workspace_ != nullptr && owned_moe_kernel_ != nullptr &&
               validateBindings("MoEGPUCurrentBatchLLEPStage::isGraphCapturable");
    }

    bool MoEGPUCurrentBatchLLEPStage::prepareGraphLaunch(
        IDeviceContext *ctx,
        void *stream)
    {
        if (!ctx || !stream)
        {
            LOG_ERROR("[MoEGPUCurrentBatchLLEPStage] Capture preparation requires an exact context and non-null stream");
            return false;
        }
        setGPUStream(stream);
        if (!bound_workspace_ ||
            !validateBindings("MoEGPUCurrentBatchLLEPStage::prepareGraphLaunch"))
        {
            return false;
        }
        IMoEKernel *kernel = ensureKernel();
        if (!kernel)
            return false;

        bindStageStream(kernel);
        if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel))
            consumer->bindWorkspace(bound_workspace_);
        return true;
    }

    void MoEGPUCurrentBatchLLEPStage::invalidateKernelDynamicState()
    {
        if (!owned_moe_kernel_)
            return;
        owned_moe_kernel_->resetDynamicState();
        owned_moe_kernel_->clearGPUStreamBinding();
    }

    std::string MoEGPUCurrentBatchLLEPStage::workspaceBufferName(
        const char *base_name,
        const std::string &workspace_name)
    {
        return MoEDeviceRebalanceStage::workspaceBufferName(
            base_name,
            workspace_name.empty()
                ? std::string("moe_current_batch_llep")
                : workspace_name);
    }

    std::string MoEGPUCurrentBatchLLEPStage::layerPublicationBufferName(
        const char *base_name,
        const std::string &workspace_name,
        int layer_idx)
    {
        if (layer_idx < 0)
        {
            throw std::invalid_argument(
                "GPU current-batch LLEP publication requires a concrete layer");
        }
        std::string identity =
            workspace_name.empty()
                ? std::string("moe_current_batch_llep")
                : workspace_name;
        identity += ":publication_layer=" + std::to_string(layer_idx);
        return MoEDeviceRebalanceStage::workspaceBufferName(
            base_name,
            identity);
    }

    std::uint32_t MoEGPUCurrentBatchLLEPStage::planCapacity(
        const Params &params)
    {
        const std::uint32_t captured_slots =
            std::min(
                params.payload_slot_capacity,
                params.local_transfer_slot_count);
        return std::max<std::uint32_t>(
            1u,
            boundedU32(
                deviceMoEPrefillLLEPMergedPlanCapacity(
                    params.config,
                    deviceMoERebalanceCommandPlanCapacity(
                        params.config,
                        params.transfer_mode),
                    captured_slots)));
    }

    std::uint32_t MoEGPUCurrentBatchLLEPStage::payloadSlotCount(
        const Params &params)
    {
        return std::min(
            std::min(
                params.payload_slot_capacity,
                params.local_transfer_slot_count),
            planCapacity(params));
    }

    std::size_t MoEGPUCurrentBatchLLEPStage::localPayloadBytes(
        const Params &params)
    {
        return static_cast<std::size_t>(payloadSlotCount(params)) *
               static_cast<std::size_t>(params.payload_slot_bytes);
    }

    DeviceMoERebalanceConfig
    MoEGPUCurrentBatchLLEPStage::transactionConfig() const
    {
        /* A current-batch placement consumes this request's evidence. The
         * shared policy helper makes histogram reset part of the immutable
         * apply transaction rather than a host-side cleanup after replay. */
        return prefillLLEPTransferConfig(
            params_.config,
            PrefillLLEPTransferPurpose::CurrentBatchMovement);
    }

    least_loaded_ep::LeastLoadedExpertAssignmentConfig
    MoEGPUCurrentBatchLLEPStage::plannerConfig() const
    {
        least_loaded_ep::LeastLoadedExpertAssignmentConfig planner;
        planner.expert_count = static_cast<std::uint32_t>(params_.num_experts);
        planner.participant_count = params_.config.participant_count;
        planner.alpha_numerator =
            std::max<std::uint32_t>(1u, params_.config.llep_alpha_numerator);
        planner.alpha_denominator =
            std::max<std::uint32_t>(1u, params_.config.llep_alpha_denominator);
        planner.lambda_numerator =
            std::max<std::uint32_t>(1u, params_.config.llep_lambda_numerator);
        planner.lambda_denominator =
            std::max<std::uint32_t>(1u, params_.config.llep_lambda_denominator);
        planner.enable_balanced_skip =
            params_.config.llep_enable_balanced_skip != 0u;
        planner.min_spread_improvement =
            params_.config.min_load_spread_improvement;
        planner.min_spread_improvement_divisor =
            params_.config.min_load_spread_improvement_divisor;
        planner.min_spread_improvement_per_critical_path_slot =
            params_.config.min_wave_spread_improvement_per_payload_slot;
        planner.min_foreign_rows_per_critical_path_slot =
            params_.config.min_foreign_rows_per_critical_path_payload_slot;

        /* A plan that cannot fit in persistent payload slots is invalid. The
         * planner receives that physical limit before it makes any placement
         * choice, so the later transport phase never needs a truncation path. */
        planner.max_weight_transfers = payloadSlotCount(params_);
        planner.max_non_owner_experts_per_participant =
            payloadSlotCount(params_);
        return planner;
    }

    bool MoEGPUCurrentBatchLLEPStage::validateBindings(
        const char *context) const
    {
        const char *label = context ? context : "MoEGPUCurrentBatchLLEPStage";
        if (!params_.device_id.is_gpu() || !params_.tp_ctx ||
            !params_.moe_runtime_table || params_.tp_device_idx < 0 ||
            params_.layer_idx < 0 || params_.seq_len <= 1 ||
            params_.num_experts <= 0 || params_.top_k <= 0 ||
            !params_.local_transfer_slots ||
            params_.local_transfer_slot_count == 0 ||
            params_.payload_slot_bytes == 0 ||
            params_.payload_slot_capacity == 0 ||
            params_.workspace_name.empty())
        {
            LOG_ERROR("[" << label << "] Incomplete GPU current-batch LLEP binding"
                      << " device=" << params_.device_id.to_string()
                      << " layer=" << params_.layer_idx
                      << " seq_len=" << params_.seq_len);
            return false;
        }
        if (params_.phase == GPUCurrentBatchLLEPPhase::PlanAndPack &&
            (!params_.routing_indices || !params_.routing_weights))
        {
            LOG_ERROR("[" << label << "] PlanAndPack requires router output tensors");
            return false;
        }
        if (params_.transfer_mode !=
            DeviceMoERebalanceTransferMode::CompactTransferSlots)
        {
            LOG_ERROR("[" << label << "] Current-batch LLEP requires compact transfer slots");
            return false;
        }
        const DeviceMoERebalanceConfig config = transactionConfig();
        if (!validateDeviceMoERebalanceConfig(config) ||
            params_.tp_ctx->degree() <= 1 ||
            params_.tp_ctx->degree() !=
                static_cast<int>(config.participant_count) ||
            params_.tp_device_idx != static_cast<int>(config.participant_id) ||
            payloadSlotCount(params_) == 0)
        {
            LOG_ERROR("[" << label << "] LLEP topology/config identity mismatch");
            return false;
        }
        auto *table = dynamic_cast<DeviceMoERuntimeTable *>(
            params_.moe_runtime_table);
        if (!table || !table->isMirroredToDevice() ||
            !table->usesOverlayEpochTicket() ||
            table->overlayPlacementSource() == nullptr ||
            params_.layer_idx >= table->layerCount() ||
            table->expertCount() != params_.num_experts ||
            table->topK() != params_.top_k)
        {
            LOG_ERROR("[" << label << "] LLEP requires a ticketed request-local child runtime table");
            return false;
        }
        return true;
    }

    IMoEKernel *MoEGPUCurrentBatchLLEPStage::ensureKernel()
    {
        if (!owned_moe_kernel_)
            owned_moe_kernel_ = KernelFactory::createMoEKernel(params_.device_id);
        if (!owned_moe_kernel_)
        {
            LOG_ERROR("[MoEGPUCurrentBatchLLEPStage] Could not create MoE kernel for "
                      << params_.device_id.to_string());
            return nullptr;
        }
        bindStageStream(owned_moe_kernel_.get());
        if (bound_workspace_)
        {
            if (auto *consumer =
                    dynamic_cast<IWorkspaceConsumer *>(owned_moe_kernel_.get()))
            {
                consumer->bindWorkspace(bound_workspace_);
            }
        }
        return owned_moe_kernel_.get();
    }

    bool MoEGPUCurrentBatchLLEPStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "MoEGPUCurrentBatchLLEPStage") ||
            !bound_workspace_ ||
            !validateBindings("MoEGPUCurrentBatchLLEPStage::execute"))
        {
            return false;
        }
        (void)requireGPUStream();
        IMoEKernel *kernel = ensureKernel();
        if (!kernel)
            return false;

        const bool ok =
            params_.phase == GPUCurrentBatchLLEPPhase::PlanAndPack
                ? executePlanAndPack(kernel)
                : executeUnpackApplyAndAssign(kernel);
        if (ok)
        {
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "gpu_current_batch_llep_graph_phase_calls",
                1.0,
                "prefill",
                params_.device_id.to_string(),
                {{"stage", params_.stage_name},
                 {"phase", phaseName(params_.phase)},
                 {"layer", std::to_string(params_.layer_idx)},
                 {"payload_bytes", std::to_string(localPayloadBytes(params_))},
                 {"collective_owner", "shared_expert_sideband"}});
        }
        return ok;
    }

    bool MoEGPUCurrentBatchLLEPStage::executePlanAndPack(IMoEKernel *kernel)
    {
        auto *runtime_layer =
            params_.moe_runtime_table->deviceLayerState(params_.layer_idx);
        auto *runtime_layers =
            params_.moe_runtime_table->deviceLayerState(0);
        if (!runtime_layer || !runtime_layers)
            return false;

        const auto plan_capacity = planCapacity(params_);
        const auto payload_slots = payloadSlotCount(params_);
        const DeviceMoERebalanceConfig config = transactionConfig();
        const MoEKernelLaunchContext launch{
            .stream = requireGPUStream(),
            .workspace = bound_workspace_,
        };

        auto *plan_entries = static_cast<DeviceMoERebalancePlanEntry *>(
            bound_workspace_->getBuffer(workspaceBufferName(
                MoEDeviceRebalanceStage::WS_TRANSFER_PLAN,
                params_.workspace_name)));
        auto *plan_count = static_cast<std::uint32_t *>(
            bound_workspace_->getBuffer(workspaceBufferName(
                MoEDeviceRebalanceStage::WS_TRANSFER_PLAN_COUNT,
                params_.workspace_name)));
        auto *command_header =
            static_cast<DeviceMoERebalanceCommandBufferHeader *>(
                bound_workspace_->getBuffer(workspaceBufferName(
                    MoEDeviceRebalanceStage::WS_COMMAND_HEADER,
                    params_.workspace_name)));
        auto *gathered_plan = static_cast<DeviceMoERebalancePlanEntry *>(
            bound_workspace_->getBuffer(workspaceBufferName(
                MoEDeviceRebalanceStage::WS_GATHERED_TRANSFER_PLAN,
                params_.workspace_name)));
        auto *gathered_headers =
            static_cast<DeviceMoERebalanceCommandBufferHeader *>(
                bound_workspace_->getBuffer(workspaceBufferName(
                    MoEDeviceRebalanceStage::WS_GATHERED_COMMAND_HEADER,
                    params_.workspace_name)));
        auto *claim_index = static_cast<DeviceMoETransferSlotClaimIndex *>(
            bound_workspace_->getBuffer(workspaceBufferName(
                MoEDeviceRebalanceStage::WS_TRANSFER_SLOT_CLAIM_INDEX,
                params_.workspace_name)));
        auto *status = static_cast<DeviceMoERebalanceStatus *>(
            bound_workspace_->getBuffer(layerPublicationBufferName(
                MoEDeviceRebalanceStage::WS_STATUS,
                params_.workspace_name,
                params_.layer_idx)));
        auto *apply_status = static_cast<DeviceMoERebalanceApplyStatus *>(
            bound_workspace_->getBuffer(layerPublicationBufferName(
                MoEDeviceRebalanceStage::WS_APPLY_STATUS,
                params_.workspace_name,
                params_.layer_idx)));
        auto *source_descriptors =
            static_cast<DeviceMoEExpertDirectoryEntry *>(
                bound_workspace_->getBuffer(workspaceBufferName(
                    MoEDeviceRebalanceStage::WS_LOCAL_SOURCE_DESCRIPTORS,
                    params_.workspace_name)));
        auto *local_payload = static_cast<std::uint8_t *>(
            bound_workspace_->getBuffer(workspaceBufferName(
                MoEDeviceRebalanceStage::WS_LOCAL_TRANSFER_PAYLOAD,
                params_.workspace_name)));
        if (!plan_entries || !plan_count || !command_header ||
            !gathered_plan || !gathered_headers || !claim_index || !status ||
            !apply_status || !source_descriptors || !local_payload)
        {
            LOG_ERROR("[MoEGPUCurrentBatchLLEPStage] PlanAndPack workspace is incomplete");
            return false;
        }

        /* Router output is replicated across this homogeneous LocalTP domain.
         * Every participant therefore constructs the same route-only ledger and
         * deterministic planner result without a metadata collective. */
        if (!kernel->groupPrefillRoutes(
                runtime_layer,
                params_.routing_indices,
                params_.routing_weights,
                params_.seq_len,
                params_.seq_len,
                params_.num_experts,
                params_.top_k,
                /*filter_to_local_runtime_experts=*/false,
                /*retain_routes_for_deferred_commit=*/false) ||
            !kernel->planPrefillRoutesLeastLoadedCurrentBatch(
                launch,
                runtime_layer,
                params_.seq_len,
                params_.seq_len,
                params_.num_experts,
                params_.top_k,
                plannerConfig()) ||
            !kernel->materializePrefillLeastLoadedMirroredDomainCommands(
                launch,
                runtime_layer,
                gathered_plan,
                gathered_headers,
                plan_capacity,
                status,
                config,
                payload_slots,
                static_cast<std::uint32_t>(params_.layer_idx)) ||
            !kernel->projectPrefillLeastLoadedDomainCommands(
                launch,
                gathered_plan,
                gathered_headers,
                plan_capacity,
                plan_entries,
                plan_count,
                command_header,
                config,
                status,
                payload_slots,
                runtime_layers,
                params_.local_transfer_slots,
                params_.local_transfer_slot_count,
                claim_index,
                /*command_buffer_count=*/1) ||
            !kernel->packDeviceRebalanceSourceDescriptors(
                launch,
                runtime_layers,
                plan_entries,
                command_header,
                plan_capacity,
                source_descriptors,
                config,
                nullptr,
                /*command_buffer_count=*/1) ||
            !kernel->packDeviceRebalanceCompactPayloads(
                launch,
                plan_entries,
                command_header,
                plan_capacity,
                source_descriptors,
                local_payload,
                payload_slots,
                params_.payload_slot_bytes,
                config,
                apply_status,
                nullptr,
                /*command_buffer_count=*/1))
        {
            LOG_ERROR("[MoEGPUCurrentBatchLLEPStage] PlanAndPack kernel transaction failed"
                      << " layer=" << params_.layer_idx);
            return false;
        }
        return true;
    }

    bool MoEGPUCurrentBatchLLEPStage::executeUnpackApplyAndAssign(
        IMoEKernel *kernel)
    {
        auto *runtime_layer =
            params_.moe_runtime_table->deviceLayerState(params_.layer_idx);
        auto *runtime_layers =
            params_.moe_runtime_table->deviceLayerState(0);
        if (!runtime_layer || !runtime_layers)
            return false;

        const auto plan_capacity = planCapacity(params_);
        const auto payload_slots = payloadSlotCount(params_);
        const DeviceMoERebalanceConfig config = transactionConfig();
        const MoEKernelLaunchContext launch{
            .stream = requireGPUStream(),
            .workspace = bound_workspace_,
        };

        auto *plan_entries = static_cast<DeviceMoERebalancePlanEntry *>(
            bound_workspace_->getBuffer(workspaceBufferName(
                MoEDeviceRebalanceStage::WS_TRANSFER_PLAN,
                params_.workspace_name)));
        auto *plan_count = static_cast<std::uint32_t *>(
            bound_workspace_->getBuffer(workspaceBufferName(
                MoEDeviceRebalanceStage::WS_TRANSFER_PLAN_COUNT,
                params_.workspace_name)));
        auto *command_header =
            static_cast<DeviceMoERebalanceCommandBufferHeader *>(
                bound_workspace_->getBuffer(workspaceBufferName(
                    MoEDeviceRebalanceStage::WS_COMMAND_HEADER,
                    params_.workspace_name)));
        auto *status = static_cast<DeviceMoERebalanceStatus *>(
            bound_workspace_->getBuffer(layerPublicationBufferName(
                MoEDeviceRebalanceStage::WS_STATUS,
                params_.workspace_name,
                params_.layer_idx)));
        auto *apply_status = static_cast<DeviceMoERebalanceApplyStatus *>(
            bound_workspace_->getBuffer(layerPublicationBufferName(
                MoEDeviceRebalanceStage::WS_APPLY_STATUS,
                params_.workspace_name,
                params_.layer_idx)));
        auto *gathered_payload = static_cast<std::uint8_t *>(
            bound_workspace_->getBuffer(workspaceBufferName(
                MoEDeviceRebalanceStage::WS_GATHERED_TRANSFER_PAYLOAD,
                params_.workspace_name)));
        if (!plan_entries || !plan_count || !command_header || !status ||
            !apply_status || !gathered_payload)
        {
            LOG_ERROR("[MoEGPUCurrentBatchLLEPStage] Unpack/apply workspace is incomplete");
            return false;
        }

        /* The TP anchor has completed on this exact stream before the graph
         * enters this node. Unpack can therefore consume the gathered lane
         * directly; no event, auxiliary stream, or host-visible ticket exists. */
        if (!kernel->unpackDeviceRebalanceCollectivePayloads(
                launch,
                plan_entries,
                plan_count,
                plan_capacity,
                command_header,
                gathered_payload,
                payload_slots,
                params_.payload_slot_bytes,
                params_.local_transfer_slots,
                params_.local_transfer_slot_count,
                config,
                apply_status,
                nullptr,
                /*command_buffer_count=*/1) ||
            !kernel->applyDeviceRebalanceArrivals(
                launch,
                runtime_layers,
                plan_entries,
                plan_count,
                plan_capacity,
                params_.local_transfer_slots,
                params_.local_transfer_slot_count,
                config,
                apply_status,
                command_header,
                params_.layer_idx) ||
            !kernel->assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers(
                launch,
                runtime_layer,
                params_.seq_len,
                params_.seq_len,
                params_.num_experts,
                params_.top_k,
                status,
                apply_status))
        {
            LOG_ERROR("[MoEGPUCurrentBatchLLEPStage] Unpack/apply/assign transaction failed"
                      << " layer=" << params_.layer_idx);
            return false;
        }
        return true;
    }

    WorkspaceRequirements
    MoEGPUCurrentBatchLLEPStage::getWorkspaceRequirements(
        int m,
        int n,
        int k) const
    {
        (void)m;
        (void)n;
        (void)k;
        WorkspaceRequirements reqs;
        const std::size_t plan_capacity = planCapacity(params_);
        const std::size_t participant_count =
            static_cast<std::size_t>(
                std::max<std::uint32_t>(1u, params_.config.participant_count));
        const std::size_t local_payload_bytes = localPayloadBytes(params_);

        /* Both phases declare the complete shared transaction BOM. The
         * workspace allocator deduplicates identical names and sizes, giving
         * plan, collective, and apply one stable set of captured addresses. */
        reqs.buffers.push_back({workspaceBufferName(
                                    MoEDeviceRebalanceStage::WS_TRANSFER_PLAN,
                                    params_.workspace_name),
                                plan_capacity * sizeof(DeviceMoERebalancePlanEntry),
                                256,
                                true});
        reqs.buffers.push_back({workspaceBufferName(
                                    MoEDeviceRebalanceStage::WS_TRANSFER_PLAN_COUNT,
                                    params_.workspace_name),
                                sizeof(std::uint32_t),
                                256,
                                true});
        reqs.buffers.push_back({workspaceBufferName(
                                    MoEDeviceRebalanceStage::WS_COMMAND_HEADER,
                                    params_.workspace_name),
                                sizeof(DeviceMoERebalanceCommandBufferHeader),
                                256,
                                true});
        reqs.buffers.push_back({workspaceBufferName(
                                    MoEDeviceRebalanceStage::WS_GATHERED_TRANSFER_PLAN,
                                    params_.workspace_name),
                                participant_count * plan_capacity *
                                    sizeof(DeviceMoERebalancePlanEntry),
                                256,
                                true});
        reqs.buffers.push_back({workspaceBufferName(
                                    MoEDeviceRebalanceStage::WS_GATHERED_COMMAND_HEADER,
                                    params_.workspace_name),
                                participant_count *
                                    sizeof(DeviceMoERebalanceCommandBufferHeader),
                                256,
                                true});
        reqs.buffers.push_back({workspaceBufferName(
                                    MoEDeviceRebalanceStage::WS_TRANSFER_SLOT_CLAIM_INDEX,
                                    params_.workspace_name),
                                deviceMoETransferSlotClaimIndexBytes(
                                    params_.local_transfer_slot_count),
                                256,
                                true});
        reqs.buffers.push_back({layerPublicationBufferName(
                                    MoEDeviceRebalanceStage::WS_STATUS,
                                    params_.workspace_name,
                                    params_.layer_idx),
                                sizeof(DeviceMoERebalanceStatus),
                                256,
                                true});
        reqs.buffers.push_back({layerPublicationBufferName(
                                    MoEDeviceRebalanceStage::WS_APPLY_STATUS,
                                    params_.workspace_name,
                                    params_.layer_idx),
                                sizeof(DeviceMoERebalanceApplyStatus),
                                256,
                                true});
        reqs.buffers.push_back({workspaceBufferName(
                                    MoEDeviceRebalanceStage::WS_LOCAL_SOURCE_DESCRIPTORS,
                                    params_.workspace_name),
                                participant_count * plan_capacity *
                                    sizeof(DeviceMoEExpertDirectoryEntry),
                                256,
                                true});
        reqs.buffers.push_back({workspaceBufferName(
                                    MoEDeviceRebalanceStage::WS_LOCAL_TRANSFER_PAYLOAD,
                                    params_.workspace_name),
                                local_payload_bytes,
                                256,
                                true});
        reqs.buffers.push_back({workspaceBufferName(
                                    MoEDeviceRebalanceStage::WS_GATHERED_TRANSFER_PAYLOAD,
                                    params_.workspace_name),
                                participant_count * local_payload_bytes,
                                256,
                                true});
        return reqs;
    }

    void MoEGPUCurrentBatchLLEPStage::bindWorkspace(
        DeviceWorkspaceManager *workspace)
    {
        bound_workspace_ = workspace;
        if (owned_moe_kernel_)
        {
            if (auto *consumer =
                    dynamic_cast<IWorkspaceConsumer *>(owned_moe_kernel_.get()))
            {
                consumer->bindWorkspace(workspace);
            }
        }
    }

    void MoEGPUCurrentBatchLLEPStage::unbindWorkspace()
    {
        if (owned_moe_kernel_)
        {
            if (auto *consumer =
                    dynamic_cast<IWorkspaceConsumer *>(owned_moe_kernel_.get()))
            {
                consumer->unbindWorkspace();
            }
        }
        bound_workspace_ = nullptr;
    }

    StageBufferRequirements
    MoEGPUCurrentBatchLLEPStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.phase == GPUCurrentBatchLLEPPhase::PlanAndPack)
        {
            if (params_.routing_indices)
            {
                reqs.addInput(
                    "routing_indices",
                    params_.routing_indices->shape(),
                    toBufferTensorType(params_.routing_indices->native_type()));
            }
            if (params_.routing_weights)
            {
                reqs.addInput(
                    "routing_weights",
                    params_.routing_weights->shape(),
                    toBufferTensorType(params_.routing_weights->native_type()));
            }
        }
        return reqs;
    }

    StageBufferContract MoEGPUCurrentBatchLLEPStage::bufferContract() const
    {
        if (params_.phase != GPUCurrentBatchLLEPPhase::PlanAndPack)
            return {};
        auto contract = StageBufferContract::build();
        if (params_.routing_indices_buffer_id.has_value())
            contract.addInput(*params_.routing_indices_buffer_id);
        if (params_.routing_weights_buffer_id.has_value())
            contract.addInput(*params_.routing_weights_buffer_id);
        return contract;
    }

    StageDumpInfo MoEGPUCurrentBatchLLEPStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.phase == GPUCurrentBatchLLEPPhase::PlanAndPack)
        {
            if (params_.routing_indices)
            {
                info.addInput(
                    "routing_indices",
                    params_.routing_indices,
                    static_cast<std::size_t>(params_.seq_len),
                    static_cast<std::size_t>(params_.top_k));
            }
            if (params_.routing_weights)
            {
                info.addInput(
                    "routing_weights",
                    params_.routing_weights,
                    static_cast<std::size_t>(params_.seq_len),
                    static_cast<std::size_t>(params_.top_k));
            }
        }
        info.addScalarInt("phase", static_cast<int>(params_.phase))
            .addScalarInt("layer_idx", params_.layer_idx)
            .addScalarInt("seq_len", params_.seq_len)
            .addScalarInt("num_experts", params_.num_experts)
            .addScalarInt("top_k", params_.top_k)
            .addScalarInt("participant_id", params_.tp_device_idx)
            .addScalarInt(
                "participant_count",
                static_cast<int>(params_.config.participant_count))
            .addScalarInt(
                "plan_capacity",
                static_cast<int>(planCapacity(params_)))
            .addScalarInt(
                "payload_slot_count",
                static_cast<int>(payloadSlotCount(params_)))
            .addScalarInt(
                "payload_bytes",
                static_cast<int>(localPayloadBytes(params_)));
        return info;
    }

} // namespace llaminar2
