/**
 * @file CUDAMoEKernel.cpp
 * @brief CUDAMoEKernel lifecycle, IQ grid table initialization, workspace binding, and router gate cache.
 */
#include "CUDAMoEKernelInternal.h"

namespace llaminar2::cudamoe_detail
{
    /**
     * @brief Publish CUDA MoE IQ constants once per native runtime generation.
     *
     * `cudaDeviceReset()` invalidates translation-unit constant memory while
     * leaving host statics intact. Keying only by ordinal would therefore make
     * the next JIT-loaded model launch with an uninitialized codebook. The
     * backend generation is the exact primary-context identity that owns these
     * tables and is shared by every supported IQ codebook (11 through 17).
     */
    bool ensureCudaMoEIQGridTablesInitialized(int device_ordinal)
    {
        static std::mutex iq_table_mutex;
        static std::unordered_map<int, std::uint64_t>
            iq_grid_runtime_generations;

        std::lock_guard<std::mutex> lock(iq_table_mutex);
        llaminar2::IBackend *const backend =
            llaminar2::getCUDABackend();
        const std::uint64_t runtime_generation = backend
                                                     ? backend->deviceRuntimeGeneration(
                                                           device_ordinal)
                                                     : 0u;
        if (runtime_generation == 0u)
        {
            LOG_ERROR(
                "[CUDAMoEKernel] IQ grid initialization has no live CUDA "
                "runtime generation for device "
                << device_ordinal);
            return false;
        }

        const auto initialized =
            iq_grid_runtime_generations.find(device_ordinal);
        if (initialized != iq_grid_runtime_generations.end() &&
            initialized->second == runtime_generation)
        {
            return true;
        }
        if (!backend->setDevice(device_ordinal))
        {
            LOG_ERROR(
                "[CUDAMoEKernel] Could not select CUDA device "
                << device_ordinal
                << " before IQ grid table publication");
            return false;
        }
        if (!cudaNativeQuantisedInitIQGridTables_tuned())
        {
            LOG_ERROR(
                "[CUDAMoEKernel] IQ grid table publication failed for CUDA:"
                << device_ordinal << " runtime_generation="
                << runtime_generation);
            return false;
        }

        iq_grid_runtime_generations[device_ordinal] = runtime_generation;
        return true;
    }
} // namespace llaminar2::cudamoe_detail

extern "C" {
    /**
     * @brief Query the capture-time CUDA production-MoE prefill policy.
     *
     * This host-only inspection shares the exact production selector and does
     * no CUDA work. Integration tests use it to prove exact-overlay precedence
     * and total generic dispatch for unseen positive M values.
     *
     * @param gateup_codebook_id Uniform gate/up execution codebook.
     * @param down_codebook_id Uniform down execution codebook.
     * @param m Number of token rows embedded in the captured graph.
     * @param hidden_size Model hidden width.
     * @param expert_width Routed expert intermediate width.
     * @param expert_count Number of routed experts.
     * @param top_k Number of selected routes per token.
     * @param gateup_tile_n Receives the ordered gate/up block width.
     * @param exact_overlay Receives one only when an exact overlay supplied the
     *        selected width; explicit trainer overrides report zero.
     * @return true when the selected policy names a compiled launch geometry.
     */
    bool cudaMoE_grouped_prefill_query_production_config(
        uint8_t gateup_codebook_id,
        uint8_t down_codebook_id,
        int m,
        int hidden_size,
        int expert_width,
        int expert_count,
        int top_k,
        int *gateup_tile_n,
        int *exact_overlay)
    {
        if (gateup_codebook_id >= 32 || down_codebook_id >= 32 ||
            !gateup_tile_n || !exact_overlay)
        {
            return false;
        }
        const CUDAMoEProductionPrefillPolicy policy =
            selectCUDAMoEProductionPrefillPolicy(
                llaminar2::MoEPrefillMeasuredOperation::CompleteFullWidthTokenRowsAllRoutes,
                uint32_t{1} << gateup_codebook_id,
                uint32_t{1} << down_codebook_id,
                m,
                hidden_size,
                expert_width,
                expert_count,
                top_k,
                /*use_gateup_kpart=*/true);
        if (!validCUDAMoEGateUpOrderedTileN(policy.gateup_tile_n))
            return false;
        *gateup_tile_n = policy.gateup_tile_n;
        *exact_overlay =
            std::strcmp(policy.source, "exact_overlay") == 0 ? 1 : 0;
        return true;
    }
}

namespace llaminar2
{

    bool CUDAMoEKernel::beginOriginalMoEOverlayInvocation(
        const MoEKernelLaunchContext &context,
        const MoEOverlayOriginalNativeBeginLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(context, "beginOriginalMoEOverlayInvocation");
        return stream && packet.valid() &&
            cudaMoEOverlayOriginalNativeBegin(&packet, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::retireOriginalMoEOverlayInvocation(
        const MoEKernelLaunchContext &context,
        const MoEOverlayOriginalNativeRetireLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(context, "retireOriginalMoEOverlayInvocation");
        return stream && packet.valid() &&
            cudaMoEOverlayOriginalNativeRetire(&packet, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::retireOriginalMoEOverlayInvocationBatch(
        const MoEKernelLaunchContext &context,
        const MoEOverlayActivationSingleRowReturnBatchLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(context, "retireOriginalMoEOverlayInvocationBatch");
        return stream && packet.valid() &&
            cudaMoEOverlayOriginalNativeRetireSingleRowBatch(&packet, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::retireOriginalMoEOverlayInvocationBatch(
        const MoEKernelLaunchContext &context,
        const MoEOverlayActivationMultiRowReturnBatchLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(context, "retireOriginalMoEOverlayInvocationBatch");
        return stream && packet.valid() &&
            cudaMoEOverlayOriginalNativeRetireMultiRowBatch(&packet, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::packMoEOverlayActivationDispatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationDispatchPackLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "packMoEOverlayActivationDispatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[CUDAMoEKernel::packMoEOverlayActivationDispatch] complete packet binding and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayActivationPackDispatch(
            &packet, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::consumeMoEOverlayActivationDispatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationDispatchConsumeLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeMoEOverlayActivationDispatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[CUDAMoEKernel::consumeMoEOverlayActivationDispatch] complete packet binding and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayActivationConsumeDispatch(
            &packet, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::packMoEOverlayActivationReturn(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationReturnPackLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "packMoEOverlayActivationReturn");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[CUDAMoEKernel::packMoEOverlayActivationReturn] complete packet binding and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayActivationPackReturn(
            &packet, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::consumeMoEOverlayActivationReturn(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationReturnConsumeLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeMoEOverlayActivationReturn");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[CUDAMoEKernel::consumeMoEOverlayActivationReturn] complete packet binding and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayActivationConsumeReturn(
            &packet, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::consumeMoEOverlayCanonicalRouteTicket(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayCanonicalRouteTicketConsumeLaunch &ticket)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeMoEOverlayCanonicalRouteTicket");
        if (!stream || !ticket.valid())
        {
            LOG_ERROR("[CUDAMoEKernel::consumeMoEOverlayCanonicalRouteTicket] complete mapped ticket binding and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayConsumeCanonicalRouteTicket(
            &ticket, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::packSingleRowMoEOverlayActivationDispatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationSingleRowDispatchPackLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "packSingleRowMoEOverlayActivationDispatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[CUDAMoEKernel::packSingleRowMoEOverlayActivationDispatch] complete one-row packet binding and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayActivationPackSingleRowDispatch(
            &packet, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::consumeSingleRowMoEOverlayActivationDispatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationSingleRowDispatchConsumeLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeSingleRowMoEOverlayActivationDispatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[CUDAMoEKernel::consumeSingleRowMoEOverlayActivationDispatch] complete one-row packet binding and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayActivationConsumeSingleRowDispatch(
            &packet, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::packSingleRowMoEOverlayActivationReturn(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationSingleRowReturnPackLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "packSingleRowMoEOverlayActivationReturn");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[CUDAMoEKernel::packSingleRowMoEOverlayActivationReturn] complete one-row packet binding and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayActivationPackSingleRowReturn(
            &packet, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::consumeSingleRowMoEOverlayActivationReturn(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationSingleRowReturnConsumeLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeSingleRowMoEOverlayActivationReturn");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[CUDAMoEKernel::consumeSingleRowMoEOverlayActivationReturn] complete one-row packet binding and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayActivationConsumeSingleRowReturn(
            &packet, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::packSingleRowMoEOverlayActivationDispatchBatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationSingleRowDispatchBatchLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "packSingleRowMoEOverlayActivationDispatchBatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[CUDAMoEKernel::packSingleRowMoEOverlayActivationDispatchBatch] persistent topology batch and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayActivationPackSingleRowDispatchBatch(
            &packet, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::consumeSingleRowMoEOverlayActivationReturnBatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationSingleRowReturnBatchLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeSingleRowMoEOverlayActivationReturnBatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[CUDAMoEKernel::consumeSingleRowMoEOverlayActivationReturnBatch] persistent topology batch, gather scratch, and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayActivationConsumeSingleRowReturnBatch(
            &packet, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::consumeMultiRowMoEOverlayActivationReturnBatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationMultiRowReturnBatchLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeMultiRowMoEOverlayActivationReturnBatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[CUDAMoEKernel::consumeMultiRowMoEOverlayActivationReturnBatch] persistent topology batch, row-index scratch, and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayActivationConsumeMultiRowReturnBatch(
            &packet, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::publishNodeLocalCanonicalRoutes(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalRoutePublishLaunch &publication)
    {
        void *stream = explicitMoELaunchStream(
            launch, "publishNodeLocalCanonicalRoutes");
        if (!stream || !publication.valid())
        {
            LOG_ERROR("[CUDAMoEKernel::publishNodeLocalCanonicalRoutes] complete mapped lane, route assignment, canonical contribution, and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayPublishNodeLocalCanonicalRoutes(
            &publication, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::acquireNodeLocalCanonicalRoutes(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalRouteConsumeLaunch &consumption)
    {
        void *stream = explicitMoELaunchStream(
            launch, "acquireNodeLocalCanonicalRoutes");
        if (!stream || !consumption.valid())
        {
            LOG_ERROR("[CUDAMoEKernel::acquireNodeLocalCanonicalRoutes] persistent peer bindings, route assignment, validation storage, output, and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayAcquireNodeLocalCanonicalRoutes(
            &consumption, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::stageNodeLocalCanonicalRoutes(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalRouteConsumeLaunch &consumption)
    {
        void *stream = explicitMoELaunchStream(
            launch, "stageNodeLocalCanonicalRoutes");
        if (!stream || !consumption.valid())
        {
            LOG_ERROR("[CUDAMoEKernel::stageNodeLocalCanonicalRoutes] mapped sources, root scratch, route assignment, and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayStageNodeLocalCanonicalRoutes(
            &consumption, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::foldNodeLocalCanonicalRoutes(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalRouteConsumeLaunch &consumption)
    {
        void *stream = explicitMoELaunchStream(
            launch, "foldNodeLocalCanonicalRoutes");
        if (!stream || !consumption.valid())
        {
            LOG_ERROR("[CUDAMoEKernel::foldNodeLocalCanonicalRoutes] persistent peer bindings, staged payload, route assignment, validation storage, output, and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayFoldNodeLocalCanonicalRoutes(
            &consumption, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::consumeSingleRowNodeLocalCanonicalRoutes(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalRouteConsumeLaunch &consumption)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeSingleRowNodeLocalCanonicalRoutes");
        if (!stream || !consumption.valid() ||
            consumption.physical_rows != 1u ||
            consumption.top_k > 32u || consumption.peer_count > 8u)
        {
            LOG_ERROR("[CUDAMoEKernel::consumeSingleRowNodeLocalCanonicalRoutes] single-row persistent peer bindings, route assignment, validation storage, output, and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayConsumeSingleRowNodeLocalCanonicalRoutes(
            &consumption, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::beginNodeLocalDensePublication(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalDensePublicationLaunch &publication)
    {
        void *stream = explicitMoELaunchStream(
            launch, "beginNodeLocalDensePublication");
        if (!stream || !publication.valid() ||
            !publication.binding.isRoot())
        {
            LOG_ERROR("[CUDAMoEKernel::beginNodeLocalDensePublication] root binding, bounded payload, and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayBeginNodeLocalDensePublication(
            &publication, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::finishNodeLocalDensePublication(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalDensePublicationLaunch &publication)
    {
        void *stream = explicitMoELaunchStream(
            launch, "finishNodeLocalDensePublication");
        if (!stream || !publication.valid() ||
            !publication.binding.isRoot())
        {
            LOG_ERROR("[CUDAMoEKernel::finishNodeLocalDensePublication] root binding, bounded payload, and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayFinishNodeLocalDensePublication(
            &publication, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::beginNodeLocalDensePublicationConsume(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalDensePublicationLaunch &publication)
    {
        void *stream = explicitMoELaunchStream(
            launch, "beginNodeLocalDensePublicationConsume");
        if (!stream || !publication.valid() || publication.binding.isRoot())
        {
            LOG_ERROR("[CUDAMoEKernel::beginNodeLocalDensePublicationConsume] peer binding, bounded payload, and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayBeginNodeLocalDensePublicationConsume(
            &publication, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::finishNodeLocalDensePublicationConsume(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalDensePublicationLaunch &publication)
    {
        void *stream = explicitMoELaunchStream(
            launch, "finishNodeLocalDensePublicationConsume");
        if (!stream || !publication.valid() || publication.binding.isRoot())
        {
            LOG_ERROR("[CUDAMoEKernel::finishNodeLocalDensePublicationConsume] peer binding, bounded payload, and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayFinishNodeLocalDensePublicationConsume(
            &publication, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::runMoEOverlayDeviceControllerAction(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayDeviceControllerActionLaunch &action)
    {
        void *stream = explicitMoELaunchStream(
            launch, "runMoEOverlayDeviceControllerAction");
        if (!stream || !action.valid())
        {
            LOG_ERROR("[CUDAMoEKernel::runMoEOverlayDeviceControllerAction] a valid mapped controller action and explicit CUDA stream are required");
            return false;
        }
        return cudaMoEOverlayRunDeviceControllerAction(
            &action, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::beginMoEOverlayServiceTelemetry(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayServiceTelemetrySample *sample)
    {
        void *stream = explicitMoELaunchStream(
            launch, "beginMoEOverlayServiceTelemetry");
        if (!stream || !sample)
            return false;
        return cudaMoEOverlayBeginServiceTelemetry(
            sample, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::finishMoEOverlayServiceTelemetry(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layer,
        DeviceMoEOverlayServiceTelemetryCell *layer_telemetry,
        DeviceMoEOverlayServiceTelemetrySample *sample,
        std::uint32_t num_experts,
        MoEOverlayServicePhaseHint hint,
        const MoEOverlayInferenceGraphRole *runtime_graph_role)
    {
        void *stream = explicitMoELaunchStream(
            launch, "finishMoEOverlayServiceTelemetry");
        if (!stream || !runtime_layer || !layer_telemetry || !sample ||
            num_experts == 0u)
        {
            return false;
        }
        return cudaMoEOverlayFinishServiceTelemetry(
            runtime_layer,
            layer_telemetry,
            sample,
            num_experts,
            static_cast<std::uint32_t>(hint),
            runtime_graph_role,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::publishMoEOverlayServiceTelemetry(
        const MoEKernelLaunchContext &launch,
        const DeviceMoEOverlayServiceTelemetryCell *telemetry,
        const DeviceMoEOverlayServiceTelemetrySample *samples,
        std::uint32_t layer_count,
        std::int32_t participant_id,
        MoEOverlayDeviceServiceTelemetryPublicationHeader *publication)
    {
        void *stream = explicitMoELaunchStream(
            launch, "publishMoEOverlayServiceTelemetry");
        if (!stream || !telemetry || !samples || !publication ||
            layer_count == 0u ||
            participant_id < 0)
        {
            return false;
        }
        return cudaMoEOverlayPublishServiceTelemetry(
            telemetry,
            samples,
            layer_count,
            participant_id,
            publication,
            device_ordinal_,
            stream);
    }

    CUDAMoEKernel::CUDAMoEKernel(int device_ordinal)
        : device_ordinal_(device_ordinal)
    {
        setMoEDevice(device_ordinal_, "CUDAMoEKernel::CUDAMoEKernel");
    }

    CUDAMoEKernel::~CUDAMoEKernel()
    {
        releaseDeviceBuffers();
    }

    void CUDAMoEKernel::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        // Workspace rebinding is common during graph warmup/capture because
        // multiple MoE stages share this singleton kernel.  A same-workspace
        // rebind must be a no-op: runtime pointer arrays are graph-capture
        // metadata, and clearing them here would erase the tables populated
        // by the warmup pass before the capture pass can consume them.
        //
        // Use DeviceWorkspaceManager::id(), not the host pointer alone.  The
        // allocator can destroy and recreate a manager at the same host address
        // while growing graph workspace requirements; treating that ABA pointer
        // reuse as a no-op would preserve stale device sub-buffer pointers.
        const uint64_t next_workspace_id = workspace ? workspace->id() : 0;
        if (workspace_ == workspace && bound_workspace_id_ == next_workspace_id)
        {
            const bool needs_descriptor_rebind =
                workspace_ &&
                (std::any_of(grouped_down_desc_tables_.begin(), grouped_down_desc_tables_.end(),
                             [](const GroupedDownDescriptorTable &table)
                             {
                                 return table.valid && !table.host_descs.empty() && !table.device_descs;
                             }) ||
                 std::any_of(grouped_gateup_desc_tables_.begin(), grouped_gateup_desc_tables_.end(),
                             [](const GroupedGateUpDescriptorTable &table)
                             {
                                 return table.valid &&
                                        !table.host_gate_descs.empty() &&
                                        !table.host_up_descs.empty() &&
                                        (!table.device_gate_descs || !table.device_up_descs);
                             }));
            if (needs_descriptor_rebind)
                (void)rebindGroupedDescriptorTablesToWorkspace("bindWorkspace");
            return;
        }

        CUDAKernelBase::bindWorkspace(workspace);
        bound_workspace_id_ = next_workspace_id;
        clearWorkspaceScratchBindings();
        if (workspace_)
            (void)rebindGroupedDescriptorTablesToWorkspace("bindWorkspace");
    }

    bool CUDAMoEKernel::bindWorkspaceBuffer(
        void **ptr,
        const char *name,
        size_t bytes,
        const char *context)
    {
        if (!ptr || !name || bytes == 0)
            return false;
        if (!workspace_)
        {
            LOG_ERROR("[CUDAMoEKernel] " << context
                                         << " requires graph-owned MoE workspace");
            return false;
        }
        const DeviceId expected = deviceId();
        if (workspace_->device() != expected)
        {
            LOG_ERROR("[CUDAMoEKernel] " << context << " requires " << expected.to_string()
                                         << " workspace, got " << workspace_->device().to_string()
                                         << " (workspace id=" << workspace_->id() << ")");
            return false;
        }
        void *buffer = workspace_->getBuffer(name);
        const size_t available = workspace_->getBufferSize(name);
        if (!buffer || available < bytes)
        {
            LOG_ERROR("[CUDAMoEKernel] " << context << " missing required workspace buffer '"
                                         << name << "' (need " << bytes << " bytes, have "
                                         << available << ")");
            return false;
        }
        if (!requireCudaDevicePointer(buffer, device_ordinal_, name, context, nullptr))
            return false;
        *ptr = buffer;
        scratch_workspace_bound_ = true;
        return true;
    }

    bool CUDAMoEKernel::bindGroupedDescriptorTableSlot(
        const char *buffer_name,
        std::size_t slot,
        int num_experts,
        DeviceNativeQuantisedMatrixDesc **device_descs,
        const char *context)
    {
        if (!buffer_name || !device_descs || num_experts <= 0 ||
            slot >= static_cast<std::size_t>(
                        MoEWorkspaceBuffers::kGroupedDescriptorTableSlots))
        {
            return false;
        }

        void *base = nullptr;
        if (!bindWorkspaceBuffer(
                &base,
                buffer_name,
                sizeof(DeviceNativeQuantisedMatrixDesc),
                context))
        {
            return false;
        }

        const std::size_t available = workspace_->getBufferSize(buffer_name);
        const std::size_t slot_bytes =
            available /
            static_cast<std::size_t>(
                MoEWorkspaceBuffers::kGroupedDescriptorTableSlots);
        const std::size_t descriptor_stride =
            slot_bytes / sizeof(DeviceNativeQuantisedMatrixDesc);
        if (descriptor_stride < static_cast<std::size_t>(num_experts))
        {
            LOG_ERROR("[CUDAMoEKernel] " << context
                                           << " descriptor slot is too narrow in '"
                                           << buffer_name << "': slot=" << slot
                                           << " stride=" << descriptor_stride
                                           << " requested_experts=" << num_experts
                                           << " workspace_bytes=" << available);
            return false;
        }

        const std::size_t descriptor_offset = slot * descriptor_stride;
        const std::size_t descriptor_capacity =
            available / sizeof(DeviceNativeQuantisedMatrixDesc);
        if (descriptor_offset + static_cast<std::size_t>(num_experts) >
            descriptor_capacity)
        {
            LOG_ERROR("[CUDAMoEKernel] " << context
                                           << " descriptor slot exceeds workspace capacity in '"
                                           << buffer_name << "': offset=" << descriptor_offset
                                           << " requested_experts=" << num_experts
                                           << " capacity=" << descriptor_capacity);
            return false;
        }

        *device_descs =
            static_cast<DeviceNativeQuantisedMatrixDesc *>(base) + descriptor_offset;
        return true;
    }

    void CUDAMoEKernel::clearWorkspaceScratchBindings() noexcept
    {
        if(original_m1_hidden_q8_read_)
            original_m1_hidden_q8_read_lifecycle_=OriginalM1HiddenQ8ReadLifecycle::Invalidated;
        original_m1_hidden_q8_read_.reset();
        grouped_prefill_grouping_.invalidate();
        runtime_gate_plan_.reset();
        d_staging_indices_ = nullptr;
        d_staging_weights_ = nullptr;
        staging_capacity_ = 0;
        d_route_logits_ = nullptr;
        route_logits_capacity_ = 0;
        route_logits_workspace_bound_ = false;
        d_group_int_indices_ = nullptr;
        d_group_offsets_ = nullptr;
        d_group_counts_ = nullptr;
        d_group_token_indices_ = nullptr;
        d_group_original_to_grouped_ = nullptr;
        d_group_original_expert_ids_ = nullptr;
        d_group_write_heads_ = nullptr;
        d_group_weights_ = nullptr;
        d_group_active_expert_ids_ = nullptr;
        d_group_expert_mask_ = nullptr;
        group_expert_mask_workspace_lease_.reset();
        group_active_expert_slots_ = 0;
        group_slots_cap_ = 0;
        group_experts_cap_ = 0;
        group_buffers_workspace_bound_ = false;
        group_expert_mask_cap_ = 0;
        group_expert_mask_hash_ = 0;
        group_expert_mask_num_experts_ = 0;
        group_expert_mask_active_experts_ = 0;
        group_expert_mask_published_ = false;
        d_prefill_A_int8_ = nullptr;
        d_prefill_A_scales_ = nullptr;
        d_prefill_swiglu_int8_ = nullptr;
        d_prefill_swiglu_scales_ = nullptr;
        d_prefill_gate_ = nullptr;
        d_prefill_up_ = nullptr;
        d_prefill_imma_directory_ = nullptr;
        prefill_imma_directory_entries_cap_ = 0;
        prefill_slots_cap_ = 0;
        prefill_d_model_cap_ = 0;
        prefill_intermediate_cap_ = 0;
        d_decode_hidden_int8_ = nullptr;
        d_decode_hidden_scales_ = nullptr;
        invalidateRouterQ8HiddenPublication();
        decode_gateup_topk_cap_ = 0;
        decode_gateup_d_model_cap_ = 0;
        decode_hidden_rows_cap_ = 0;
        d_grouped_gateup_gate_partials_ = nullptr;
        d_grouped_gateup_up_partials_ = nullptr;
        grouped_gateup_kpart_active_cap_ = 0;
        grouped_gateup_kpart_partitions_cap_ = 0;
        grouped_gateup_kpart_intermediate_cap_ = 0;
        d_grouped_down_partials_ = nullptr;
        grouped_down_kpart_partitions_cap_ = 0;
        grouped_down_kpart_d_model_cap_ = 0;
        grouped_down_kpart_slots_cap_ = 0;
        d_decode_swiglu_int8_ = nullptr;
        d_decode_swiglu_scales_ = nullptr;
        decode_down_topk_cap_ = 0;
        decode_down_intermediate_cap_ = 0;
        d_routing_decode_expert_ids_ = nullptr;
        routing_decode_metadata_cap_ = 0;
        fixed_gateup_metadata_states_.clear();
        fixed_down_metadata_states_.clear();
        runtime_pointer_workspace_owners_.reset();
        gateup_pointer_slot_ready_.fill(false);
        down_pointer_slot_ready_.fill(false);
        for (auto &table : grouped_down_desc_tables_)
        {
            table.device_descs = nullptr;
            table.device_floating_descs = nullptr;
            table.workspace_publication.reset();
            table.workspace_slot = 0;
        }
        for (auto &table : grouped_gateup_desc_tables_)
        {
            table.device_gate_descs = nullptr;
            table.device_up_descs = nullptr;
            table.device_floating_gate_descs = nullptr;
            table.device_floating_up_descs = nullptr;
            table.workspace_publication.reset();
            table.workspace_slot = 0;
        }
        router_q8_gate_cache_.clear();
        scratch_workspace_bound_ = false;
    }

    bool CUDAMoEKernel::publishGroupedDownDescriptorTable(
        GroupedDownDescriptorTable &table,
        const char *context)
    {
        const bool floating = deviceMoEWeightFormatIsFloating(
            table.weight_format);
        if (!workspace_ || !table.valid ||
            (floating ? table.host_floating_descs.empty()
                      : table.host_descs.empty()) ||
            table.num_experts <= 0)
        {
            LOG_ERROR("[CUDAMoEKernel] "
                      << (context ? context : "down descriptor publication")
                      << " requires a bound workspace and complete host table");
            return false;
        }
        cudaStream_t stream = static_cast<cudaStream_t>(getStream());
        if (!stream)
        {
            LOG_ERROR("[CUDAMoEKernel] "
                      << (context ? context : "down descriptor publication")
                      << " requires an explicit CUDA stream");
            return false;
        }

        // Source2 consumes live descriptors from its runtime bank. An empty
        // original template needs local initialization only: allocation E is
        // not an H2D communication extent. Keep the same slot and ready event.
        const bool empty_original_image = table.original_family.valid() &&
            (floating ? (std::all_of(table.host_floating_descs.begin(),table.host_floating_descs.end(),isBlankGroupedFloatingDesc))
                      : (std::all_of(table.host_descs.begin(),table.host_descs.end(),isBlankGroupedDesc)));

        if (floating)
        {
            const size_t desc_bytes =
                static_cast<size_t>(table.num_experts) *
                sizeof(DeviceMoEFloatingMatrixDesc);
            const auto publication_result =
                workspace_->getOrCreatePersistentPublication(
                    kCudaGroupedDownDescriptorLeaseDomain,
                    groupedFloatingDescriptorPublicationKey(
                        table.host_floating_descs.data(),
                        nullptr,
                        table.weight_format,
                        table.num_experts,
                        table.d_model,
                        table.intermediate, table.descriptor_source),
                    MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                    [&](std::size_t slot) -> std::shared_ptr<void>
                    {
                        DeviceNativeQuantisedMatrixDesc *slot_base = nullptr;
                        if (!bindGroupedDescriptorTableSlot(
                                MoEWorkspaceBuffers::CUDA_GROUPED_DOWN_DESC_TABLES,
                                slot,
                                table.num_experts,
                                &slot_base,
                                "CUDA grouped floating down descriptor publication"))
                        {
                            return {};
                        }
                        auto *device_descs =
                            reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(slot_base);

                        cudaEvent_t ready_event = nullptr;
                        cudaError_t err = cudaEventCreateWithFlags(
                            &ready_event, cudaEventDisableTiming);
                        if (err != cudaSuccess || !ready_event)
                            return {};

                        auto publication =
                            std::shared_ptr<GroupedDescriptorWorkspacePublication>(
                                new GroupedDescriptorWorkspacePublication{
                                    .ready_event = static_cast<void *>(ready_event),
                                    .primary_descs = device_descs,
                                    .secondary_descs = nullptr,
                                    .workspace_slot = slot,
                                },
                                [](GroupedDescriptorWorkspacePublication *value)
                                {
                                    if (value && value->ready_event)
                                    {
                                        (void)cudaEventDestroy(
                                            static_cast<cudaEvent_t>(value->ready_event));
                                    }
                                    delete value;
                                });
                        err = empty_original_image
                            ? cudaMemsetAsync(device_descs,0,desc_bytes,stream)
                            : cudaMemcpyAsync(device_descs,table.host_floating_descs.data(),desc_bytes,cudaMemcpyHostToDevice,stream);
                        if (err != cudaSuccess)
                            return {};
                        err = cudaEventRecord(ready_event, stream);
                        if (err != cudaSuccess)
                            std::terminate();
                        return publication;
                    });
            if (!publication_result)
                return false;

            auto publication = std::static_pointer_cast<
                GroupedDescriptorWorkspacePublication>(
                publication_result.publication);
            const cudaError_t wait_err = cudaStreamWaitEvent(
                stream,
                static_cast<cudaEvent_t>(publication->ready_event),
                0);
            if (wait_err != cudaSuccess)
                return false;

            table.device_descs = nullptr;
            table.device_floating_descs =
                static_cast<DeviceMoEFloatingMatrixDesc *>(
                    publication->primary_descs);
            table.workspace_publication = std::move(publication);
            table.workspace_slot = publication_result.slot;
            return true;
        }

        const size_t desc_bytes =
            static_cast<size_t>(table.num_experts) *
            sizeof(DeviceNativeQuantisedMatrixDesc);
        const auto publication_result =
            workspace_->getOrCreatePersistentPublication(
                kCudaGroupedDownDescriptorLeaseDomain,
                groupedDescriptorPublicationKey(
                    table.host_descs.data(),
                    nullptr,
                    table.num_experts,
                    table.d_model,
                    table.intermediate, table.descriptor_source),
                MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                [&](std::size_t slot) -> std::shared_ptr<void>
                {
                    DeviceNativeQuantisedMatrixDesc *device_descs = nullptr;
                    if (!bindGroupedDescriptorTableSlot(
                            MoEWorkspaceBuffers::
                                CUDA_GROUPED_DOWN_DESC_TABLES,
                            slot,
                            table.num_experts,
                            &device_descs,
                            "CUDA grouped down descriptor publication"))
                    {
                        return {};
                    }

                    cudaEvent_t ready_event = nullptr;
                    cudaError_t err = cudaEventCreateWithFlags(
                        &ready_event,
                        cudaEventDisableTiming);
                    if (err != cudaSuccess || !ready_event)
                    {
                        LOG_ERROR("[CUDAMoEKernel] Failed to create grouped "
                                  "down descriptor readiness event: "
                                  << cudaGetErrorString(err));
                        return {};
                    }
                    auto publication =
                        std::shared_ptr<
                            GroupedDescriptorWorkspacePublication>(
                            new GroupedDescriptorWorkspacePublication{
                                .ready_event =
                                    static_cast<void *>(ready_event),
                                .primary_descs = device_descs,
                                .secondary_descs = nullptr,
                                .workspace_slot = slot,
                            },
                            [](GroupedDescriptorWorkspacePublication *value)
                            {
                                if (value && value->ready_event)
                                {
                                    (void)cudaEventDestroy(
                                        static_cast<cudaEvent_t>(
                                            value->ready_event));
                                }
                                delete value;
                            });

                    err = empty_original_image
                            ? cudaMemsetAsync(device_descs,0,desc_bytes,stream)
                            : cudaMemcpyAsync(device_descs,table.host_descs.data(),desc_bytes,cudaMemcpyHostToDevice,stream);
                    if (err != cudaSuccess)
                    {
                        LOG_ERROR("[CUDAMoEKernel] Grouped down descriptor "
                                  "H2D publication failed: "
                                  << cudaGetErrorString(err));
                        return {};
                    }
                    err = cudaEventRecord(ready_event, stream);
                    if (err != cudaSuccess)
                    {
                        LOG_ERROR("[CUDAMoEKernel] Grouped down descriptor "
                                  "publication cannot record readiness: "
                                  << cudaGetErrorString(err));
                        /*
                         * The H2D producer has already been submitted. Without
                         * its event there is no coherent slot rollback, and a
                         * stream/device sync would be a hidden recovery path.
                         */
                        std::terminate();
                    }
                    return publication;
                });
        if (!publication_result)
        {
            LOG_ERROR("[CUDAMoEKernel] Failed to publish grouped down "
                      "descriptor table");
            return false;
        }

        auto publication =
            std::static_pointer_cast<
                GroupedDescriptorWorkspacePublication>(
                publication_result.publication);
        const cudaError_t wait_err = cudaStreamWaitEvent(
            stream,
            static_cast<cudaEvent_t>(publication->ready_event),
            0);
        if (wait_err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] Failed to adopt grouped down "
                      "descriptor publication: "
                      << cudaGetErrorString(wait_err));
            return false;
        }

        table.device_descs = static_cast<DeviceNativeQuantisedMatrixDesc *>(
            publication->primary_descs);
        table.device_floating_descs = nullptr;
        table.workspace_publication = std::move(publication);
        table.workspace_slot = publication_result.slot;
        return true;
    }

    bool CUDAMoEKernel::publishGroupedGateUpDescriptorTable(
        GroupedGateUpDescriptorTable &table,
        const char *context)
    {
        const bool floating = deviceMoEWeightFormatIsFloating(
            table.weight_format);
        if (!workspace_ || !table.valid ||
            (floating ? table.host_floating_gate_descs.empty()
                      : table.host_gate_descs.empty()) ||
            (floating ? table.host_floating_up_descs.empty()
                      : table.host_up_descs.empty()) ||
            table.num_experts <= 0)
        {
            LOG_ERROR("[CUDAMoEKernel] "
                      << (context ? context : "gate/up descriptor publication")
                      << " requires a bound workspace and complete host tables");
            return false;
        }
        cudaStream_t stream = static_cast<cudaStream_t>(getStream());
        if (!stream)
        {
            LOG_ERROR("[CUDAMoEKernel] "
                      << (context ? context : "gate/up descriptor publication")
                      << " requires an explicit CUDA stream");
            return false;
        }

        // Source2 consumes live descriptors from its runtime bank. An empty
        // original template needs local initialization only: allocation E is
        // not an H2D communication extent. Keep the same slot and ready event.
        const bool empty_original_image = table.original_family.valid() &&
            (floating ? (std::all_of(table.host_floating_gate_descs.begin(),table.host_floating_gate_descs.end(),isBlankGroupedFloatingDesc) &&
               std::all_of(table.host_floating_up_descs.begin(),table.host_floating_up_descs.end(),isBlankGroupedFloatingDesc))
                      : (std::all_of(table.host_gate_descs.begin(),table.host_gate_descs.end(),isBlankGroupedDesc) &&
               std::all_of(table.host_up_descs.begin(),table.host_up_descs.end(),isBlankGroupedDesc)));

        if (floating)
        {
            const size_t desc_bytes =
                static_cast<size_t>(table.num_experts) *
                sizeof(DeviceMoEFloatingMatrixDesc);
            const auto publication_result =
                workspace_->getOrCreatePersistentPublication(
                    kCudaGroupedGateUpDescriptorLeaseDomain,
                    groupedFloatingDescriptorPublicationKey(
                        table.host_floating_gate_descs.data(),
                        table.host_floating_up_descs.data(),
                        table.weight_format,
                        table.num_experts,
                        table.d_model,
                        table.intermediate, table.descriptor_source),
                    MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                    [&](std::size_t slot) -> std::shared_ptr<void>
                    {
                        DeviceNativeQuantisedMatrixDesc *gate_slot_base = nullptr;
                        DeviceNativeQuantisedMatrixDesc *up_slot_base = nullptr;
                        if (!bindGroupedDescriptorTableSlot(
                                MoEWorkspaceBuffers::CUDA_GROUPED_GATE_DESC_TABLES,
                                slot,
                                table.num_experts,
                                &gate_slot_base,
                                "CUDA grouped floating gate descriptor publication") ||
                            !bindGroupedDescriptorTableSlot(
                                MoEWorkspaceBuffers::CUDA_GROUPED_UP_DESC_TABLES,
                                slot,
                                table.num_experts,
                                &up_slot_base,
                                "CUDA grouped floating up descriptor publication"))
                        {
                            return {};
                        }
                        auto *device_gate_descs =
                            reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                                gate_slot_base);
                        auto *device_up_descs =
                            reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                                up_slot_base);

                        cudaEvent_t ready_event = nullptr;
                        cudaError_t err = cudaEventCreateWithFlags(
                            &ready_event, cudaEventDisableTiming);
                        if (err != cudaSuccess || !ready_event)
                            return {};
                        auto publication =
                            std::shared_ptr<GroupedDescriptorWorkspacePublication>(
                                new GroupedDescriptorWorkspacePublication{
                                    .ready_event = static_cast<void *>(ready_event),
                                    .primary_descs = device_gate_descs,
                                    .secondary_descs = device_up_descs,
                                    .workspace_slot = slot,
                                },
                                [](GroupedDescriptorWorkspacePublication *value)
                                {
                                    if (value && value->ready_event)
                                    {
                                        (void)cudaEventDestroy(
                                            static_cast<cudaEvent_t>(value->ready_event));
                                    }
                                    delete value;
                                });
                        err = empty_original_image
                            ? cudaMemsetAsync(device_gate_descs,0,desc_bytes,stream)
                            : cudaMemcpyAsync(device_gate_descs,table.host_floating_gate_descs.data(),desc_bytes,cudaMemcpyHostToDevice,stream);
                        if (err != cudaSuccess)
                            return {};
                        err = empty_original_image
                            ? cudaMemsetAsync(device_up_descs,0,desc_bytes,stream)
                            : cudaMemcpyAsync(device_up_descs,table.host_floating_up_descs.data(),desc_bytes,cudaMemcpyHostToDevice,stream);
                        if (err != cudaSuccess)
                            std::terminate();
                        err = cudaEventRecord(ready_event, stream);
                        if (err != cudaSuccess)
                            std::terminate();
                        return publication;
                    });
            if (!publication_result)
                return false;
            auto publication = std::static_pointer_cast<
                GroupedDescriptorWorkspacePublication>(
                publication_result.publication);
            const cudaError_t wait_err = cudaStreamWaitEvent(
                stream,
                static_cast<cudaEvent_t>(publication->ready_event),
                0);
            if (wait_err != cudaSuccess)
                return false;

            table.device_gate_descs = nullptr;
            table.device_up_descs = nullptr;
            table.device_floating_gate_descs =
                static_cast<DeviceMoEFloatingMatrixDesc *>(
                    publication->primary_descs);
            table.device_floating_up_descs =
                static_cast<DeviceMoEFloatingMatrixDesc *>(
                    publication->secondary_descs);
            table.workspace_publication = std::move(publication);
            table.workspace_slot = publication_result.slot;
            return true;
        }

        const size_t desc_bytes =
            static_cast<size_t>(table.num_experts) *
            sizeof(DeviceNativeQuantisedMatrixDesc);
        const auto publication_result =
            workspace_->getOrCreatePersistentPublication(
                kCudaGroupedGateUpDescriptorLeaseDomain,
                groupedDescriptorPublicationKey(
                    table.host_gate_descs.data(),
                    table.host_up_descs.data(),
                    table.num_experts,
                    table.d_model,
                    table.intermediate, table.descriptor_source),
                MoEWorkspaceBuffers::kGroupedDescriptorTableSlots,
                [&](std::size_t slot) -> std::shared_ptr<void>
                {
                    DeviceNativeQuantisedMatrixDesc *device_gate_descs =
                        nullptr;
                    DeviceNativeQuantisedMatrixDesc *device_up_descs =
                        nullptr;
                    if (!bindGroupedDescriptorTableSlot(
                            MoEWorkspaceBuffers::
                                CUDA_GROUPED_GATE_DESC_TABLES,
                            slot,
                            table.num_experts,
                            &device_gate_descs,
                            "CUDA grouped gate descriptor publication") ||
                        !bindGroupedDescriptorTableSlot(
                            MoEWorkspaceBuffers::
                                CUDA_GROUPED_UP_DESC_TABLES,
                            slot,
                            table.num_experts,
                            &device_up_descs,
                            "CUDA grouped up descriptor publication"))
                    {
                        return {};
                    }

                    cudaEvent_t ready_event = nullptr;
                    cudaError_t err = cudaEventCreateWithFlags(
                        &ready_event,
                        cudaEventDisableTiming);
                    if (err != cudaSuccess || !ready_event)
                    {
                        LOG_ERROR("[CUDAMoEKernel] Failed to create grouped "
                                  "gate/up descriptor readiness event: "
                                  << cudaGetErrorString(err));
                        return {};
                    }
                    auto publication =
                        std::shared_ptr<
                            GroupedDescriptorWorkspacePublication>(
                            new GroupedDescriptorWorkspacePublication{
                                .ready_event =
                                    static_cast<void *>(ready_event),
                                .primary_descs = device_gate_descs,
                                .secondary_descs = device_up_descs,
                                .workspace_slot = slot,
                            },
                            [](GroupedDescriptorWorkspacePublication *value)
                            {
                                if (value && value->ready_event)
                                {
                                    (void)cudaEventDestroy(
                                        static_cast<cudaEvent_t>(
                                            value->ready_event));
                                }
                                delete value;
                            });

                    err = empty_original_image
                            ? cudaMemsetAsync(device_gate_descs,0,desc_bytes,stream)
                            : cudaMemcpyAsync(device_gate_descs,table.host_gate_descs.data(),desc_bytes,cudaMemcpyHostToDevice,stream);
                    if (err != cudaSuccess)
                    {
                        LOG_ERROR("[CUDAMoEKernel] Grouped gate descriptor "
                                  "H2D publication failed: "
                                  << cudaGetErrorString(err));
                        return {};
                    }
                    err = empty_original_image
                            ? cudaMemsetAsync(device_up_descs,0,desc_bytes,stream)
                            : cudaMemcpyAsync(device_up_descs,table.host_up_descs.data(),desc_bytes,cudaMemcpyHostToDevice,stream);
                    if (err != cudaSuccess)
                    {
                        LOG_ERROR("[CUDAMoEKernel] Grouped up descriptor H2D "
                                  "publication failed after gate submission: "
                                  << cudaGetErrorString(err));
                        std::terminate();
                    }
                    err = cudaEventRecord(ready_event, stream);
                    if (err != cudaSuccess)
                    {
                        LOG_ERROR("[CUDAMoEKernel] Grouped gate/up descriptor "
                                  "publication cannot record readiness: "
                                  << cudaGetErrorString(err));
                        std::terminate();
                    }
                    return publication;
                });
        if (!publication_result)
        {
            LOG_ERROR("[CUDAMoEKernel] Failed to publish grouped gate/up "
                      "descriptor table");
            return false;
        }

        auto publication =
            std::static_pointer_cast<
                GroupedDescriptorWorkspacePublication>(
                publication_result.publication);
        const cudaError_t wait_err = cudaStreamWaitEvent(
            stream,
            static_cast<cudaEvent_t>(publication->ready_event),
            0);
        if (wait_err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel] Failed to adopt grouped gate/up "
                      "descriptor publication: "
                      << cudaGetErrorString(wait_err));
            return false;
        }

        table.device_gate_descs = static_cast<DeviceNativeQuantisedMatrixDesc *>(
            publication->primary_descs);
        table.device_up_descs = static_cast<DeviceNativeQuantisedMatrixDesc *>(
            publication->secondary_descs);
        table.device_floating_gate_descs = nullptr;
        table.device_floating_up_descs = nullptr;
        table.workspace_publication = std::move(publication);
        table.workspace_slot = publication_result.slot;
        return true;
    }

    bool CUDAMoEKernel::rebindGroupedDescriptorTablesToWorkspace(const char *context)
    {
        if (grouped_down_desc_tables_.empty() && grouped_gateup_desc_tables_.empty())
            return true;
        if (rejectCudaPersistentMetadataMutationDuringCapture(
                getStream(), "rebind grouped descriptor tables"))
        {
            return false;
        }
        if (!setMoEDevice(device_ordinal_, context ? context : "rebindGroupedDescriptorTablesToWorkspace"))
            return false;
        for (auto &table : grouped_down_desc_tables_)
        {
            const bool floating = deviceMoEWeightFormatIsFloating(
                table.weight_format);
            if (!table.valid ||
                (floating ? table.host_floating_descs.empty()
                          : table.host_descs.empty()) ||
                table.num_experts <= 0)
                continue;
            if (!publishGroupedDownDescriptorTable(table, context))
                return false;
        }

        for (auto &table : grouped_gateup_desc_tables_)
        {
            const bool floating = deviceMoEWeightFormatIsFloating(
                table.weight_format);
            if (!table.valid ||
                (floating ? table.host_floating_gate_descs.empty()
                          : table.host_gate_descs.empty()) ||
                (floating ? table.host_floating_up_descs.empty()
                          : table.host_up_descs.empty()) ||
                table.num_experts <= 0)
            {
                continue;
            }
            if (!publishGroupedGateUpDescriptorTable(table, context))
                return false;
        }
        return true;
    }

    void CUDAMoEKernel::resetDynamicState()
    {
        grouped_prefill_grouping_.invalidate();
        runtime_gate_plan_.reset();
        /*
         * This hook is reached only on hard kernel-dynamic reset boundaries.
         * Replay-preserving request reset deliberately avoids
         * KernelFactory::resetAllDynamicState(), because captured CUDA graphs
         * may still reference workspace-backed pointer tables.  Once the hard
         * reset path is chosen, grouped prefill/decode scratch, mask upload
         * hashes, router caches, and descriptor-table handles must all be
         * treated as invalid so cached stages rebuild them before the next
         * eager warmup or capture.
         */
        clearWorkspaceScratchBindings();
        grouped_down_desc_tables_.clear();
        grouped_gateup_desc_tables_.clear();
    }

    bool CUDAMoEKernel::runMoEOverlayEpochParentAction(
        const MoEKernelLaunchContext &launch,DeviceMoEOverlayEpochControl *control,
        DeviceMoEOverlayEpochStatus *status,DeviceMoEOverlayEpochParentAction action)
    {
        void *stream=explicitMoELaunchStream(launch,"runMoEOverlayEpochParentAction");
        if(!stream || !control || !status)return false;
        return cudaMoEOverlayEpochParentAction(control,status,action,device_ordinal_,stream);
    }

    bool CUDAMoEKernel::acquireMoEOverlayEpoch(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        DeviceMoEOverlayEpochTicket *ticket,
        DeviceMoEOverlayEpochStatus *status,
        const std::uint64_t *external_admission_epoch,
        DeviceMoEOverlayEpochAdmissionBarrierBinding admission_barrier,
        MoEOverlayPeerPlacementEpochBinding peer_placement_epoch)
    {
        void *stream = explicitMoELaunchStream(
            launch, "acquireMoEOverlayEpoch");
        if (!stream || !control || !ticket || !status)
        {
            LOG_ERROR("[CUDAMoEKernel::acquireMoEOverlayEpoch] control, ticket, status, and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayEpochAcquire(
            control,
            ticket,
            status,
            external_admission_epoch,
            admission_barrier,
            peer_placement_epoch,
            device_ordinal_,
            stream);
    }

    bool CUDAMoEKernel::releaseMoEOverlayEpoch(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        DeviceMoEOverlayEpochTicket *ticket,
        DeviceMoEOverlayEpochStatus *status)
    {
        void *stream = explicitMoELaunchStream(
            launch, "releaseMoEOverlayEpoch");
        if (!stream || !control || !ticket || !status)
        {
            LOG_ERROR("[CUDAMoEKernel::releaseMoEOverlayEpoch] control, ticket, status, and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayEpochRelease(
            control, ticket, status, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::reserveMoEOverlayEpochCandidate(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *status)
    {
        void *stream = explicitMoELaunchStream(
            launch, "reserveMoEOverlayEpochCandidate");
        if (!stream || !control || !candidate_epoch || !status)
        {
            LOG_ERROR("[CUDAMoEKernel::reserveMoEOverlayEpochCandidate] control, device epoch, status, and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayEpochReserveCandidate(
            control, candidate_epoch, status, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::markMoEOverlayEpochCandidateReady(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *status)
    {
        void *stream = explicitMoELaunchStream(
            launch, "markMoEOverlayEpochCandidateReady");
        if (!stream || !control || !candidate_epoch || !status)
        {
            LOG_ERROR("[CUDAMoEKernel::markMoEOverlayEpochCandidateReady] control, device epoch, status, and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayEpochMarkCandidateReady(
            control, candidate_epoch, status, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::publishMoEOverlayEpochCandidate(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *status)
    {
        void *stream = explicitMoELaunchStream(
            launch, "publishMoEOverlayEpochCandidate");
        if (!stream || !control || !candidate_epoch || !status)
        {
            LOG_ERROR("[CUDAMoEKernel::publishMoEOverlayEpochCandidate] control, device epoch, status, and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayEpochPublishCandidate(
            control, candidate_epoch, status, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::abortMoEOverlayEpochCandidate(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *status)
    {
        void *stream = explicitMoELaunchStream(
            launch, "abortMoEOverlayEpochCandidate");
        if (!stream || !control || !candidate_epoch || !status)
        {
            LOG_ERROR("[CUDAMoEKernel::abortMoEOverlayEpochCandidate] control, device epoch, status, and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayEpochAbortCandidate(
            control, candidate_epoch, status, device_ordinal_, stream);
    }

    bool CUDAMoEKernel::retireMoEOverlayEpoch(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *retiring_epoch,
        DeviceMoEOverlayEpochStatus *status)
    {
        void *stream = explicitMoELaunchStream(
            launch, "retireMoEOverlayEpoch");
        if (!stream || !control || !retiring_epoch || !status)
        {
            LOG_ERROR("[CUDAMoEKernel::retireMoEOverlayEpoch] control, device epoch, status, and explicit stream are required");
            return false;
        }
        return cudaMoEOverlayEpochRetire(
            control, retiring_epoch, status, device_ordinal_, stream);
    }

    void CUDAMoEKernel::releaseDeviceBuffers() noexcept
    {
        cudaSetDevice(device_ordinal_);
        clearWorkspaceScratchBindings();
        staging_capacity_ = 0;
        route_logits_capacity_ = 0;
        group_slots_cap_ = 0;
        group_experts_cap_ = 0;
        group_active_expert_slots_ = 0;
        grouped_prefill_grouping_.invalidate();
        group_expert_mask_cap_ = 0;
        group_expert_mask_hash_ = 0;
        group_expert_mask_num_experts_ = 0;
        group_expert_mask_active_experts_ = 0;
        group_expert_mask_published_ = false;
        prefill_slots_cap_ = 0;
        prefill_d_model_cap_ = 0;
        prefill_intermediate_cap_ = 0;
        decode_gateup_topk_cap_ = 0;
        decode_gateup_d_model_cap_ = 0;
        decode_hidden_rows_cap_ = 0;
        invalidateRouterQ8HiddenPublication();
        router_q8_gate_cache_.clear();
        grouped_gateup_kpart_active_cap_ = 0;
        grouped_gateup_kpart_partitions_cap_ = 0;
        grouped_gateup_kpart_intermediate_cap_ = 0;
        decode_down_topk_cap_ = 0;
        decode_down_intermediate_cap_ = 0;
        grouped_down_kpart_partitions_cap_ = 0;
        grouped_down_kpart_d_model_cap_ = 0;
        grouped_down_kpart_slots_cap_ = 0;
        grouped_down_desc_tables_.clear();
        grouped_gateup_desc_tables_.clear();
        gateup_pointer_slot_ready_.fill(false);
        down_pointer_slot_ready_.fill(false);

    }

    bool CUDAMoEKernel::ensureStagingCapacity(int count)
    {
        if (count <= staging_capacity_)
            return true;

        void *staging_indices = nullptr;
        void *staging_weights = nullptr;
        const bool ok =
            bindWorkspaceBuffer(&staging_indices, MoEWorkspaceBuffers::STAGING_INDICES,
                                static_cast<size_t>(count) * sizeof(int), "staging indices") &&
            bindWorkspaceBuffer(&staging_weights, MoEWorkspaceBuffers::STAGING_WEIGHTS,
                                static_cast<size_t>(count) * sizeof(float), "staging weights");
        if (!ok)
        {
            d_staging_indices_ = nullptr;
            d_staging_weights_ = nullptr;
            staging_capacity_ = 0;
            return false;
        }

        d_staging_indices_ = static_cast<int *>(staging_indices);
        d_staging_weights_ = static_cast<float *>(staging_weights);
        staging_capacity_ = count;
        return true;
    }

    bool CUDAMoEKernel::ensureRouteBufferCapacity(size_t logits_count)
    {
        // Capacity alone is not enough for graph-captured MoE routing.  This
        // kernel is a per-device singleton shared by many stages, so a previous
        // eager or stale binding can leave counters that look large enough while
        // the actual pointers are no longer the current graph workspace buffers.
        if (route_logits_workspace_bound_ &&
            d_route_logits_ && logits_count <= route_logits_capacity_)
        {
            return true;
        }

        void *route_logits = nullptr;
        const bool ok = bindWorkspaceBuffer(
            &route_logits,
            MoEWorkspaceBuffers::ROUTE_LOGITS,
            logits_count * sizeof(float),
            "route logits");
        if (!ok)
        {
            d_route_logits_ = nullptr;
            route_logits_capacity_ = 0;
            route_logits_workspace_bound_ = false;
            return false;
        }

        d_route_logits_ = static_cast<float *>(route_logits);
        route_logits_capacity_ = logits_count;
        route_logits_workspace_bound_ = true;
        return true;
    }

    bool CUDAMoEKernel::ensureGroupingBufferCapacity(int total_slots, int num_experts)
    {
        /*
         * Capacity describes only the shape of the last successful binding.
         * The CUDA MoE kernel is a per-device singleton shared by sibling graph
         * stages, so workspace replacement can revoke every pointer without
         * changing the requested shape. Require the complete binding proof and
         * every pointer before allowing the fast reuse path. This mirrors the
         * route-buffer contract and prevents a stale 0xff grouping memset from
         * targeting an arena range owned by another graph buffer.
         */
        if (group_buffers_workspace_bound_ &&
            d_group_int_indices_ && d_group_token_indices_ &&
            d_group_original_to_grouped_ && d_group_original_expert_ids_ &&
            d_group_weights_ && d_group_offsets_ && d_group_counts_ &&
            d_group_active_expert_ids_ && d_group_write_heads_ &&
            total_slots <= group_slots_cap_ &&
            num_experts <= group_experts_cap_)
        {
            return true;
        }

        void *group_int_indices = nullptr;
        void *group_token_indices = nullptr;
        void *group_original_to_grouped = nullptr;
        void *group_original_expert_ids = nullptr;
        void *group_weights = nullptr;
        void *group_offsets = nullptr;
        void *group_counts = nullptr;
        void *group_active_expert_ids = nullptr;
        void *group_write_heads = nullptr;

        const bool ok =
            bindWorkspaceBuffer(&group_int_indices, MoEWorkspaceBuffers::GROUP_INT_INDICES,
                                static_cast<size_t>(total_slots) * sizeof(int), "group int indices") &&
            bindWorkspaceBuffer(&group_token_indices, MoEWorkspaceBuffers::GROUP_TOKEN_INDICES,
                                static_cast<size_t>(total_slots) * sizeof(int), "group token indices") &&
            bindWorkspaceBuffer(&group_original_to_grouped, MoEWorkspaceBuffers::GROUP_ORIGINAL_TO_GROUPED,
                                static_cast<size_t>(total_slots) * sizeof(int), "group original-to-grouped") &&
            bindWorkspaceBuffer(&group_original_expert_ids, MoEWorkspaceBuffers::GROUP_ORIGINAL_EXPERT_IDS,
                                static_cast<size_t>(total_slots) * sizeof(int), "group original expert ids") &&
            bindWorkspaceBuffer(&group_weights, MoEWorkspaceBuffers::GROUP_WEIGHTS,
                                static_cast<size_t>(total_slots) * sizeof(float), "group weights") &&
            bindWorkspaceBuffer(&group_offsets, MoEWorkspaceBuffers::GROUP_OFFSETS,
                                static_cast<size_t>(num_experts) * sizeof(int), "group offsets") &&
            bindWorkspaceBuffer(&group_counts, MoEWorkspaceBuffers::GROUP_COUNTS,
                                static_cast<size_t>(num_experts) * sizeof(int), "group counts") &&
            bindWorkspaceBuffer(&group_active_expert_ids, MoEWorkspaceBuffers::GROUP_ACTIVE_EXPERT_IDS,
                                static_cast<size_t>(num_experts) * sizeof(int), "group active expert ids") &&
            bindWorkspaceBuffer(&group_write_heads, MoEWorkspaceBuffers::GROUP_WRITE_HEADS,
                                static_cast<size_t>(num_experts) * sizeof(int), "group write heads");
        if (!ok)
        {
            d_group_int_indices_ = nullptr;
            d_group_token_indices_ = nullptr;
            d_group_original_to_grouped_ = nullptr;
            d_group_original_expert_ids_ = nullptr;
            d_group_weights_ = nullptr;
            d_group_offsets_ = nullptr;
            d_group_counts_ = nullptr;
            d_group_active_expert_ids_ = nullptr;
            d_group_write_heads_ = nullptr;
            group_slots_cap_ = 0;
            group_experts_cap_ = 0;
            group_buffers_workspace_bound_ = false;
            return false;
        }

        d_group_int_indices_ = static_cast<int *>(group_int_indices);
        d_group_token_indices_ = static_cast<int *>(group_token_indices);
        d_group_original_to_grouped_ = static_cast<int *>(group_original_to_grouped);
        d_group_original_expert_ids_ = static_cast<int *>(group_original_expert_ids);
        d_group_weights_ = static_cast<float *>(group_weights);
        d_group_offsets_ = static_cast<int *>(group_offsets);
        d_group_counts_ = static_cast<int *>(group_counts);
        d_group_active_expert_ids_ = static_cast<int *>(group_active_expert_ids);
        d_group_write_heads_ = static_cast<int *>(group_write_heads);
        group_slots_cap_ = total_slots;
        group_experts_cap_ = num_experts;
        group_buffers_workspace_bound_ = true;
        return true;
    }

    bool CUDAMoEKernel::ensureGroupedPrefillScratchCapacity(int total_slots, int d_model, int intermediate)
    {
        const bool need_realloc = total_slots > prefill_slots_cap_ ||
                                  d_model > prefill_d_model_cap_ ||
                                  intermediate > prefill_intermediate_cap_;
        if (!need_realloc &&
            d_prefill_A_int8_ &&
            d_prefill_A_scales_ &&
            d_prefill_swiglu_int8_ &&
            d_prefill_swiglu_scales_ &&
            d_prefill_gate_ &&
            d_prefill_up_)
        {
            return true;
        }

        const int max_dim = std::max(d_model, intermediate);
        const int max_blocks = (max_dim + 31) / 32;
        const int intermediate_blocks = (intermediate + 31) / 32;
        void *prefill_a_int8 = nullptr;
        void *prefill_a_scales = nullptr;
        void *prefill_swiglu_int8 = nullptr;
        void *prefill_swiglu_scales = nullptr;
        void *prefill_gate = nullptr;
        void *prefill_up = nullptr;
        const bool ok =
            bindWorkspaceBuffer(&prefill_a_int8, MoEWorkspaceBuffers::PREFILL_A_INT8,
                                static_cast<size_t>(total_slots) * max_dim * sizeof(int8_t), "prefill A int8") &&
            bindWorkspaceBuffer(&prefill_a_scales, MoEWorkspaceBuffers::PREFILL_A_SCALES,
                                static_cast<size_t>(total_slots) * max_blocks * sizeof(float), "prefill A scales") &&
            bindWorkspaceBuffer(&prefill_swiglu_int8, MoEWorkspaceBuffers::PREFILL_SWIGLU_INT8,
                                static_cast<size_t>(total_slots) * intermediate * sizeof(int8_t), "prefill SwiGLU int8") &&
            bindWorkspaceBuffer(&prefill_swiglu_scales, MoEWorkspaceBuffers::PREFILL_SWIGLU_SCALES,
                                static_cast<size_t>(total_slots) * intermediate_blocks * sizeof(float), "prefill SwiGLU scales") &&
            bindWorkspaceBuffer(&prefill_gate, MoEWorkspaceBuffers::PREFILL_GATE,
                                static_cast<size_t>(total_slots) * max_dim * sizeof(float), "prefill gate") &&
            bindWorkspaceBuffer(&prefill_up, MoEWorkspaceBuffers::PREFILL_UP,
                                static_cast<size_t>(total_slots) * intermediate * sizeof(float), "prefill up");
        if (!ok)
        {
            d_prefill_A_int8_ = nullptr;
            d_prefill_A_scales_ = nullptr;
            d_prefill_swiglu_int8_ = nullptr;
            d_prefill_swiglu_scales_ = nullptr;
            d_prefill_gate_ = nullptr;
            d_prefill_up_ = nullptr;
            prefill_slots_cap_ = 0;
            prefill_d_model_cap_ = 0;
            prefill_intermediate_cap_ = 0;
            return false;
        }

        d_prefill_A_int8_ = static_cast<int8_t *>(prefill_a_int8);
        d_prefill_A_scales_ = static_cast<float *>(prefill_a_scales);
        d_prefill_swiglu_int8_ = static_cast<int8_t *>(prefill_swiglu_int8);
        d_prefill_swiglu_scales_ = static_cast<float *>(prefill_swiglu_scales);
        d_prefill_gate_ = static_cast<float *>(prefill_gate);
        d_prefill_up_ = static_cast<float *>(prefill_up);
        prefill_slots_cap_ = total_slots;
        prefill_d_model_cap_ = d_model;
        prefill_intermediate_cap_ = intermediate;
        return true;
    }

    bool CUDAMoEKernel::ensureGroupedImmaDirectoryCapacity(
        int total_slots,
        int num_experts)
    {
        if (total_slots <= 0 || num_experts <= 0 ||
            num_experts > cuda::moe::kGroupedImmaMaximumExperts)
        {
            LOG_ERROR("[CUDAMoEKernel::ensureGroupedImmaDirectoryCapacity] "
                      "invalid grouped IMMA geometry total_slots="
                      << total_slots << " num_experts=" << num_experts);
            return false;
        }

        const std::size_t required_entries =
            cuda::moe::groupedImmaDirectoryEntries(
                static_cast<std::size_t>(total_slots),
                static_cast<std::size_t>(num_experts));
        if (d_prefill_imma_directory_ &&
            prefill_imma_directory_entries_cap_ >=
                static_cast<int>(required_entries))
        {
            return true;
        }

        void *directory = nullptr;
        if (!bindWorkspaceBuffer(
                &directory,
                MoEWorkspaceBuffers::CUDA_PREFILL_WORK_DIRECTORY,
                required_entries * sizeof(uint32_t),
                "grouped IMMA work directory"))
        {
            d_prefill_imma_directory_ = nullptr;
            prefill_imma_directory_entries_cap_ = 0;
            return false;
        }

        d_prefill_imma_directory_ = static_cast<uint32_t *>(directory);
        prefill_imma_directory_entries_cap_ =
            static_cast<int>(required_entries);
        return true;
    }

    bool CUDAMoEKernel::ensureGroupedGateUpDecodeCapacity(
        int top_k,
        int d_model,
        int hidden_rows)
    {
        if (top_k <= 0 || top_k > static_cast<int>(kRuntimePointerArrayMaxTopK) ||
            d_model <= 0 || (d_model % 32) != 0 || hidden_rows <= 0)
            return false;

        const bool need_growth = top_k > decode_gateup_topk_cap_ ||
                                 d_model > decode_gateup_d_model_cap_ ||
                                 hidden_rows > decode_hidden_rows_cap_ ||
                                 !d_decode_hidden_int8_ || !d_decode_hidden_scales_;
        if (!need_growth)
            return true;

        const int blocks_per_row = d_model / 32;
        void *decode_hidden_int8 = nullptr;
        void *decode_hidden_scales = nullptr;
        const bool ok =
            bindWorkspaceBuffer(&decode_hidden_int8, MoEWorkspaceBuffers::DECODE_HIDDEN_INT8,
                                static_cast<size_t>(hidden_rows) *
                                    static_cast<size_t>(d_model) * sizeof(int8_t),
                                "decode hidden int8") &&
            bindWorkspaceBuffer(&decode_hidden_scales, MoEWorkspaceBuffers::DECODE_HIDDEN_SCALES,
                                static_cast<size_t>(hidden_rows) *
                                    static_cast<size_t>(blocks_per_row) * sizeof(float),
                                "decode hidden scales");
        if (!ok)
        {
            d_decode_hidden_int8_ = nullptr;
            d_decode_hidden_scales_ = nullptr;
            decode_gateup_topk_cap_ = 0;
            decode_gateup_d_model_cap_ = 0;
            decode_hidden_rows_cap_ = 0;
            return false;
        }

        d_decode_hidden_int8_ = static_cast<int8_t *>(decode_hidden_int8);
        d_decode_hidden_scales_ = static_cast<float *>(decode_hidden_scales);
        decode_gateup_topk_cap_ = top_k;
        decode_gateup_d_model_cap_ = d_model;
        decode_hidden_rows_cap_ = hidden_rows;
        return true;
    }

    bool CUDAMoEKernel::prepareOriginalM1HiddenQ8Publication(const TensorBase* hidden,DeviceRowRange rows)
    {
        const auto identity=rows.captureIdentity();auto& publication=router_q8_hidden_publication_;
        if(isGraphCaptureActive()||!publication||!publication->requiredOriginalM1Source()||
           hidden!=publication->requiredOriginalM1Source()||identity!=DeviceRowRange::fullyActive(1).captureIdentity()||!workspace_)
            throw std::invalid_argument("Original M1 hidden Q8 requires its cold declared fully-active source");
        const auto stream=getStream();const int width=publication->requiredOriginalM1Width();
        if(router_q8_publication_access_==MoERouterQ8PublicationAccess::ProducerAndConsumer) {
            if(!publication->original_m1_binding)
                publication->original_m1_binding=std::make_shared<OriginalM1HiddenQ8Binding>(
                    DeviceId::cuda(device_ordinal_),*hidden,width,stream,*workspace_,OriginalM1HiddenQ8Bank::OriginalCUDA);
            publication->original_m1_binding->validate(*hidden,stream,workspace_);
        } else {
            if(!publication->original_m1_binding)throw std::logic_error("Original M1 hidden Q8 producer must prepare before its required reader");
            if(original_m1_hidden_q8_read_lifecycle_==OriginalM1HiddenQ8ReadLifecycle::Invalidated)
                throw std::logic_error("Original M1 hidden Q8 reader was retired by hard scratch reset");
            if(!original_m1_hidden_q8_read_)
                original_m1_hidden_q8_read_=std::make_shared<OriginalM1HiddenQ8ReadView>(
                    publication->original_m1_binding,*hidden,stream,*workspace_);
            original_m1_hidden_q8_read_->validate(*hidden,stream,workspace_);
            original_m1_hidden_q8_read_lifecycle_=OriginalM1HiddenQ8ReadLifecycle::Prepared;
        }
        return true;
    }

    std::uint64_t CUDAMoEKernel::originalM1HiddenQ8BindingSignature() const
    {
        if(!router_q8_hidden_publication_||!router_q8_hidden_publication_->requiredOriginalM1Source())return 0;
        const auto& publication=*router_q8_hidden_publication_;
        if(publication.original_m1_binding) {
            const auto stream=ExplicitGPUStream(gpu_stream_).get();
            if(original_m1_hidden_q8_read_lifecycle_==OriginalM1HiddenQ8ReadLifecycle::Invalidated)
                throw std::logic_error("Original M1 hidden Q8 reader lost its prepared scratch lifetime");
            if(router_q8_publication_access_==MoERouterQ8PublicationAccess::ProducerAndConsumer)
                publication.original_m1_binding->validate(*publication.requiredOriginalM1Source(),stream,workspace_);
            else if(original_m1_hidden_q8_read_)
                original_m1_hidden_q8_read_->validate(*publication.requiredOriginalM1Source(),stream,workspace_);
        }
        // Declaration identity stays stable before/after first cold preparation;
        // leases independently authenticate every embedded pointer on replay.
        return 0x4d31513800000001ULL;
    }

    bool CUDAMoEKernel::prepareRequiredRouterQ8Hidden(int rows, int width, int top_k)
    {
        if (!router_q8_hidden_publication_ ||
            !router_q8_hidden_publication_->hasRequiredConsumer())
            return true;
        if (router_q8_publication_access_ != MoERouterQ8PublicationAccess::ProducerAndConsumer ||
            !router_q8_hidden_publication_->acceptsProducerRows(rows, width))
        {
            LOG_ERROR("[CUDAMoEKernel] required Q8 producer geometry/access mismatch");
            return false;
        }
        return ensureGroupedGateUpDecodeCapacity(top_k, width, rows);
    }

    bool CUDAMoEKernel::publishRequiredRouterQ8Hidden(
        const float *source, int rows, int width, int top_k, const int *live_rows)
    {
        if (!router_q8_hidden_publication_ ||
            !router_q8_hidden_publication_->hasRequiredConsumer())
            return true;
        if (!prepareRequiredRouterQ8Hidden(rows, width, top_k) ||
            !cudaMoE_publish_original_hidden_q8_rows(source, d_decode_hidden_int8_,
                d_decode_hidden_scales_, rows, width, device_ordinal_, getStream(), live_rows))
            return false;
        publishRouterQ8Hidden(source, rows, isCudaMoEDecodeCaptureActive(getStream()),
            RouterQ8Basis::Original);
        return true;
    }

    void CUDAMoEKernel::invalidateRouterQ8HiddenPublication() noexcept
    {
        if (router_q8_publication_access_ ==
                MoERouterQ8PublicationAccess::ProducerAndConsumer &&
            router_q8_hidden_publication_)
        {
            router_q8_hidden_publication_->clearPayload();
        }
    }

    bool CUDAMoEKernel::bindRouterQ8HiddenPublication(
        std::shared_ptr<MoERouterQ8HiddenPublication> publication,
        MoERouterQ8PublicationAccess access)
    {
        if (!publication)
        {
            LOG_ERROR("[CUDAMoEKernel::bindRouterQ8HiddenPublication] null publication");
            return false;
        }
        if (publication->device_ordinal >= 0 &&
            (publication->backend != DeviceType::CUDA ||
             publication->device_ordinal != device_ordinal_))
        {
            LOG_ERROR("[CUDAMoEKernel::bindRouterQ8HiddenPublication] device mismatch"
                      << " publication_backend="
                      << static_cast<int>(publication->backend)
                      << " publication_ordinal=" << publication->device_ordinal
                      << " kernel_ordinal=" << device_ordinal_);
            return false;
        }

        publication->backend = DeviceType::CUDA;
        publication->device_ordinal = device_ordinal_;
        if(router_q8_hidden_publication_&&router_q8_hidden_publication_->original_m1_binding&&
           (publication!=router_q8_hidden_publication_||access!=router_q8_publication_access_))
            throw std::logic_error("Captured original M1 hidden Q8 publication/access cannot be rebound");
        router_q8_hidden_publication_ = std::move(publication);
        router_q8_publication_access_ = access;
        return true;
    }

    void CUDAMoEKernel::publishRouterQ8Hidden(
        const float *source,
        int rows,
        bool recorded_during_capture,
        RouterQ8Basis router_basis) noexcept
    {
        if (router_q8_publication_access_ !=
                MoERouterQ8PublicationAccess::ProducerAndConsumer ||
            !router_q8_hidden_publication_)
        {
            LOG_ERROR("[CUDAMoEKernel::publishRouterQ8Hidden] consumer-only kernel attempted publication");
            return;
        }
        if (!source || rows <= 0 || rows > decode_hidden_rows_cap_ ||
            !d_decode_hidden_int8_ || !d_decode_hidden_scales_)
        {
            invalidateRouterQ8HiddenPublication();
            return;
        }

        auto &publication = *router_q8_hidden_publication_;
        publication.backend = DeviceType::CUDA;
        publication.device_ordinal = device_ordinal_;
        publication.source_rows = source;
        publication.quantized_rows = d_decode_hidden_int8_;
        publication.row_scales = d_decode_hidden_scales_;
        publication.published_rows = rows;
        publication.d_model_capacity = decode_gateup_d_model_cap_;
        publication.blocks_per_row_capacity =
            (decode_gateup_d_model_cap_ + 31) / 32;
        publication.capture_recorded = recorded_during_capture;
        if (router_basis == RouterQ8Basis::SignedHadamard32)
            PerfStatsCollector::addCounter("kernel", "cuda_moe_router_hadamard32_calls", 1.0,
                "router", "cuda:" + std::to_string(device_ordinal_),
                {{"rows", std::to_string(rows)}, {"expert_hidden_basis", "original"}, {"block", "32"}});

    }

    bool CUDAMoEKernel::canReuseRouterQ8Hidden(
        const float *source,
        int rows,
        int d_model) const
    {
        if(router_q8_hidden_publication_&&router_q8_hidden_publication_->requiredOriginalM1Source()) {
            const auto& p=*router_q8_hidden_publication_;const auto& binding=p.original_m1_binding;
            if(!binding||rows!=1||d_model!=p.requiredOriginalM1Width()||source!=binding->source())
                throw std::logic_error("Required original M1 hidden Q8 source/row geometry is absent");
            (void)originalM1HiddenQ8BindingSignature();
            if(p.backend!=DeviceType::CUDA||p.device_ordinal!=device_ordinal_||p.d_model_capacity<d_model||p.blocks_per_row_capacity<d_model/32||
               p.source_rows!=source||p.quantized_rows!=binding->codes()||p.row_scales!=binding->scales()||p.published_rows!=1||
               (p.capture_recorded&&!isGraphCaptureActive()))
                throw std::logic_error("Required original M1 hidden Q8 payload/provenance is absent");
            return true;
        }
        return routerQ8HiddenReuseBlockReason(source, rows, d_model) == nullptr;
    }

    const char *CUDAMoEKernel::routerQ8HiddenReuseBlockReason(
        const float *source,
        int rows,
        int d_model) const noexcept
    {
        if (!debugEnv().gemm.cuda_moe_reuse_router_q8_hidden)
            return "disabled";
        if (!router_q8_hidden_publication_)
            return "not_published";
        const auto &publication = *router_q8_hidden_publication_;
        if (!publication.quantized_rows || !publication.row_scales)
            return "not_published";
        if (publication.backend != DeviceType::CUDA ||
            publication.device_ordinal != device_ordinal_)
            return "device_mismatch";
        if (publication.source_rows != source)
            return "source_changed";
        if (rows <= 0 || publication.published_rows < rows)
            return "insufficient_rows";
        if (d_model <= 0 || publication.d_model_capacity < d_model ||
            publication.blocks_per_row_capacity < ((d_model + 31) / 32))
            return "insufficient_d_model_capacity";
        if (publication.capture_recorded &&
            !isCudaMoEDecodeCaptureActive(getStream()))
        {
            return "capture_provenance";
        }
        return nullptr;
    }

    const CUDAMoEKernel::RouterQ8GateCacheEntry *CUDAMoEKernel::getOrCreateQ8RouterGateCache(
        const float *gate_device_ptr,
        int d_model,
        int num_experts)
    {
        if (!debugEnv().gemm.cuda_moe_router_q8)
            return nullptr;
        if (!gate_device_ptr || d_model <= 0 || num_experts <= 0 ||
            (d_model % 32) != 0)
        {
            LOG_ERROR("[CUDAMoEKernel::getOrCreateQ8RouterGateCache] invalid Q8 router gate request "
                      "(gate_device_ptr=" << static_cast<const void *>(gate_device_ptr)
                      << " d_model=" << d_model
                      << " num_experts=" << num_experts << ")");
            return nullptr;
        }
        if (!workspace_)
        {
            LOG_ERROR("[CUDAMoEKernel::getOrCreateQ8RouterGateCache] "
                      "graph-owned workspace is required before cache publication");
            return nullptr;
        }

        const DeviceResidentRouterGateCacheKey cache_key =
            DeviceResidentRouterGateCacheKey::make(
                workspace_->id(), gate_device_ptr, d_model, num_experts,
                routerQ8Basis(debugEnv().gemm.moe_router_arithmetic));
        for (const auto &entry : router_q8_gate_cache_)
        {
            if (entry.key == cache_key &&
                entry.d_gate_weights_q8 && entry.d_gate_scales)
            {
                return &entry;
            }
        }

        void *stream = getStream();
        const bool capture_active =
            isGraphCaptureActive() ||
            (deviceContext() && deviceContext()->isDeviceGraphCaptureActive()) ||
            isCudaStreamCapturing(stream);
        if (capture_active)
        {
            LOG_ERROR("[CUDAMoEKernel::getOrCreateQ8RouterGateCache] Q8 router cache miss during graph capture");
            return nullptr;
        }
        if (!stream)
        {
            LOG_ERROR("[CUDAMoEKernel::getOrCreateQ8RouterGateCache] explicit CUDA stream is required");
            return nullptr;
        }

        const size_t d_model_sz = static_cast<size_t>(d_model);
        const size_t experts_sz = static_cast<size_t>(num_experts);
        if (d_model_sz > std::numeric_limits<size_t>::max() / experts_sz)
        {
            LOG_ERROR("[CUDAMoEKernel::getOrCreateQ8RouterGateCache] router gate size overflow");
            return nullptr;
        }
        const size_t element_count = d_model_sz * experts_sz;
        const int blocks_per_row = d_model / 32;
        const size_t scale_count = static_cast<size_t>(num_experts) * static_cast<size_t>(blocks_per_row);

        if (!setMoEDevice(device_ordinal_, "getOrCreateQ8RouterGateCache"))
            return nullptr;

        void *gate_weights_base = nullptr;
        void *gate_scales_base = nullptr;
        const size_t weights_payload_bytes = element_count * sizeof(int8_t);
        const size_t scales_payload_bytes = scale_count * sizeof(float);
        if (!bindWorkspaceBuffer(&gate_weights_base,
                                 MoEWorkspaceBuffers::CUDA_ROUTER_Q8_GATE_WEIGHTS,
                                 weights_payload_bytes,
                                 "CUDA Q8 router gate weights") ||
            !bindWorkspaceBuffer(&gate_scales_base,
                                 MoEWorkspaceBuffers::CUDA_ROUTER_Q8_GATE_SCALES,
                                 scales_payload_bytes,
                                 "CUDA Q8 router gate scales"))
        {
            return nullptr;
        }

        /*
         * Graph-local router/expert pairs intentionally own separate backend
         * kernel objects. The converted model weight does not: its bytes are
         * immutable and belong to this workspace allocation. Publish them under
         * the exact device-weight key so every graph adopts one slot instead of
         * allocating and reconverting an identical private copy.
         *
         * The manager serializes this setup-only factory. Recording the event
         * before the factory returns means another graph can enqueue a pure
         * device-side dependency without a host or device synchronization.
         */
        const auto publication_result =
            workspace_->getOrCreatePersistentPublication(
                kCudaRouterQ8GatePublicationDomain,
                routerPublicationKey(cache_key),
                MoEWorkspaceBuffers::kRouterGateCacheSlots,
                [&](std::size_t slot) -> std::shared_ptr<void>
                {
                    void *weights = workspace_->getPersistentSlotBuffer(
                        MoEWorkspaceBuffers::CUDA_ROUTER_Q8_GATE_WEIGHTS,
                        MoEWorkspaceBuffers::kRouterGateCacheSlots,
                        slot,
                        weights_payload_bytes);
                    void *scales = workspace_->getPersistentSlotBuffer(
                        MoEWorkspaceBuffers::CUDA_ROUTER_Q8_GATE_SCALES,
                        MoEWorkspaceBuffers::kRouterGateCacheSlots,
                        slot,
                        scales_payload_bytes);
                    if (!weights || !scales)
                        return {};

                    cudaEvent_t ready_event = nullptr;
                    cudaError_t event_err = cudaEventCreateWithFlags(
                        &ready_event, cudaEventDisableTiming);
                    if (event_err != cudaSuccess || !ready_event)
                    {
                        LOG_ERROR("[CUDAMoEKernel::getOrCreateQ8RouterGateCache] "
                                  "failed to create immutable publication event: "
                                  << cudaGetErrorString(event_err));
                        return {};
                    }

                    auto publication =
                        std::shared_ptr<RouterGateWorkspacePublication>(
                            new RouterGateWorkspacePublication{
                                .ready_event = static_cast<void *>(ready_event),
                                .primary_weights = weights,
                                .scales = scales,
                                .workspace_slot = slot,
                            },
                            [](RouterGateWorkspacePublication *value)
                            {
                                if (value && value->ready_event)
                                {
                                    cudaEventDestroy(
                                        static_cast<cudaEvent_t>(
                                            value->ready_event));
                                }
                                delete value;
                            });

                    if (!cudaMoE_quantize_router_gate_q8(
                            gate_device_ptr,
                            static_cast<int8_t *>(weights),
                            static_cast<float *>(scales),
                            d_model,
                            num_experts,
                            device_ordinal_,
                            stream, routerQ8Basis(debugEnv().gemm.moe_router_arithmetic)))
                    {
                        LOG_ERROR("[CUDAMoEKernel::getOrCreateQ8RouterGateCache] "
                                  "FP32->Q8 router gate conversion launch failed");
                        return {};
                    }

                    event_err = cudaEventRecord(
                        ready_event, static_cast<cudaStream_t>(stream));
                    if (event_err != cudaSuccess)
                    {
                        LOG_ERROR("[CUDAMoEKernel::getOrCreateQ8RouterGateCache] "
                                  "failed to record immutable publication event: "
                                  << cudaGetErrorString(event_err));
                        /*
                         * Device work has already been submitted. Returning to
                         * ordinary teardown would let another graph observe an
                         * unpublishable in-flight write, and synchronizing here
                         * would be a hidden recovery path. This invariant is
                         * intentionally process-fatal.
                         */
                        std::terminate();
                    }
                    return publication;
                });
        if (!publication_result)
        {
            LOG_ERROR("[CUDAMoEKernel::getOrCreateQ8RouterGateCache] "
                      "failed to publish immutable router gate");
            return nullptr;
        }

        auto publication =
            std::static_pointer_cast<RouterGateWorkspacePublication>(
                publication_result.publication);
        const cudaError_t wait_err = cudaStreamWaitEvent(
            static_cast<cudaStream_t>(stream),
            static_cast<cudaEvent_t>(publication->ready_event),
            0);
        if (wait_err != cudaSuccess)
        {
            LOG_ERROR("[CUDAMoEKernel::getOrCreateQ8RouterGateCache] "
                      "failed to adopt immutable router publication: "
                      << cudaGetErrorString(wait_err));
            return nullptr;
        }

        RouterQ8GateCacheEntry entry{};
        entry.key = cache_key;
        entry.blocks_per_row = blocks_per_row;
        entry.element_count = element_count;
        entry.scale_count = scale_count;
        entry.d_gate_weights_q8 =
            static_cast<int8_t *>(publication->primary_weights);
        entry.d_gate_scales =
            static_cast<float *>(publication->scales);
        entry.workspace_publication = std::move(publication);
        entry.workspace_slot = publication_result.slot;
        router_q8_gate_cache_.push_back(std::move(entry));

        LOG_TRACE("[CUDAMoEKernel] Adopted Q8 router gate source_device="
                  << static_cast<const void *>(gate_device_ptr)
                  << " workspace_id=" << cache_key.workspace_id
                  << " slot=" << publication_result.slot
                  << " created=" << publication_result.created
                  << " shape=[" << num_experts << "," << d_model << "] payload_bytes="
                  << element_count << " scale_bytes=" << (scale_count * sizeof(float)));
        return &router_q8_gate_cache_.back();
    }

} // namespace llaminar2
