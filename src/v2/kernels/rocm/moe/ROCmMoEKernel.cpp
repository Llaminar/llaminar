/**
 * @file ROCmMoEKernel.cpp
 * @brief ROCmMoEKernel compiled partition policy, grouped_ffn format program, lifecycle, and workspace binding.
 */
#include "ROCmMoEKernelInternal.h"

namespace llaminar2::rocm
{
/** @copydoc compiledMoERouteOwnedModelPartitionPolicy */
MoERouteOwnedModelPartitionPolicy compiledMoERouteOwnedModelPartitionPolicy() noexcept
{
#if LLAMINAR_ROCM_MOE_M1_OWNER_PARTITION_PLANES == 1
    return MoERouteOwnedModelPartitionPolicy::FourWorkerPlanes;
#elif LLAMINAR_ROCM_MOE_M1_OWNER_PARTITION_PLANES == 0
    return MoERouteOwnedModelPartitionPolicy::Serial;
#elif LLAMINAR_ROCM_MOE_M1_OWNER_PARTITION_PLANES == 2
    return MoERouteOwnedModelPartitionPolicy::Automatic;
#else
#error "invalid source-bound ROCm owner partition policy"
#endif
}
}


namespace llaminar2::rocm::grouped_ffn
{
/** @copydoc BoundSourceFormatProgram::prepare */
BoundSourceFormatProgram BoundSourceFormatProgram::prepare(
    DeviceMoERuntimeTable& table,int modelLayer,const Program& program,
    const std::array<DeviceNativeQuantisedMatrixDesc,3>& shared)
{
    const auto* source=table.fragmentDescriptorDeclaration(modelLayer);
    const auto device=table.deviceId();
    if(!source || !device.is_rocm() || !table.isMirroredToDevice() ||
       source->weightFormat()!=DeviceMoEWeightFormat::NativeQuantised ||
       !source->validFor(device,experts,hidden,routedColumns(program)))
        throw std::invalid_argument("Grouped FFN format program requires its original native ROCm runtime layer");
    const auto& original=source->sourceFormats();
    if(const auto* measured=std::get_if<MeasuredProgram>(&program)) {
        if(measuredDecoder(routedColumns(measured->geometry),source->nativeGateUpEnvelope(),
                source->nativeDownEnvelope(),original)!=measured->decoder)
            throw std::invalid_argument("Measured grouped FFN disagrees with the original runtime source");
        if(measured->sums==AffineSumSource::RouterPublication &&
           !hasMeasuredSumLayout(measured->geometry,measured->decoder,original,shared))
            throw std::invalid_argument("Measured grouped FFN runtime cannot consume the captured K32 sums");
    } else {
        const auto& format=std::get<SourceFormatProgram>(program);
        const auto matchesEnvelope=[](const auto& actual,const auto& expected) {
            return actual.execution_codebook_mask==expected.execution_codebook_mask &&
                actual.policy_codebook_mask==expected.policy_codebook_mask;
        };
        const std::array sources{original[0].native_quantised,original[1].native_quantised,
            original[2].native_quantised};
        if(sources!=format.routedSources() ||
           !matchesEnvelope(source->nativeGateUpEnvelope(),format.formats().routedGateUp) ||
           !matchesEnvelope(source->nativeDownEnvelope(),format.formats().routedDown))
            throw std::invalid_argument("Grouped FFN format program disagrees with the original runtime source");
    }
    auto* runtime=table.deviceLayerState(modelLayer);
    if(!runtime)throw std::invalid_argument("Grouped FFN format program requires admitted device storage");
    return BoundSourceFormatProgram(program,runtime,nativeQuantisedHardwareForDevice(device.ordinal));
}

/** @copydoc prepareFormatTables */
bool prepareFormatTables(const WeightFormats& formats,int device)
{
    return !groupedPrefillMaskNeedsIQTables(formats.executionMask()) || ensureIQGridTablesInitialized(device);
}
}

namespace llaminar2
{

    bool ROCmMoEKernel::beginOriginalMoEOverlayInvocation(
        const MoEKernelLaunchContext &context,
        const MoEOverlayOriginalNativeBeginLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(context, "beginOriginalMoEOverlayInvocation");
        return stream && packet.valid() &&
            hipMoEOverlayOriginalNativeBegin(&packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::retireOriginalMoEOverlayInvocation(
        const MoEKernelLaunchContext &context,
        const MoEOverlayOriginalNativeRetireLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(context, "retireOriginalMoEOverlayInvocation");
        return stream && packet.valid() &&
            hipMoEOverlayOriginalNativeRetire(&packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::retireOriginalMoEOverlayInvocationBatch(
        const MoEKernelLaunchContext &context,
        const MoEOverlayActivationSingleRowReturnBatchLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(context, "retireOriginalMoEOverlayInvocationBatch");
        return stream && packet.valid() &&
            hipMoEOverlayOriginalNativeRetireSingleRowBatch(&packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::retireOriginalMoEOverlayInvocationBatch(
        const MoEKernelLaunchContext &context,
        const MoEOverlayActivationMultiRowReturnBatchLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(context, "retireOriginalMoEOverlayInvocationBatch");
        return stream && packet.valid() &&
            hipMoEOverlayOriginalNativeRetireMultiRowBatch(&packet, device_ordinal_, stream);
    }


    bool ROCmMoEKernel::packMoEOverlayActivationDispatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationDispatchPackLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "packMoEOverlayActivationDispatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::packMoEOverlayActivationDispatch] complete packet binding and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationPackDispatch(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::consumeMoEOverlayActivationDispatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationDispatchConsumeLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeMoEOverlayActivationDispatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::consumeMoEOverlayActivationDispatch] complete packet binding and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationConsumeDispatch(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::packMoEOverlayActivationReturn(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationReturnPackLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "packMoEOverlayActivationReturn");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::packMoEOverlayActivationReturn] complete packet binding and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationPackReturn(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::consumeMoEOverlayActivationReturn(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationReturnConsumeLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeMoEOverlayActivationReturn");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::consumeMoEOverlayActivationReturn] complete packet binding and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationConsumeReturn(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::consumeMoEOverlayCanonicalRouteTicket(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayCanonicalRouteTicketConsumeLaunch &ticket)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeMoEOverlayCanonicalRouteTicket");
        if (!stream || !ticket.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::consumeMoEOverlayCanonicalRouteTicket] complete mapped ticket binding and explicit stream are required");
            return false;
        }
        return hipMoEOverlayConsumeCanonicalRouteTicket(
            &ticket, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::packSingleRowMoEOverlayActivationDispatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationSingleRowDispatchPackLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "packSingleRowMoEOverlayActivationDispatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::packSingleRowMoEOverlayActivationDispatch] complete one-row packet binding and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationPackSingleRowDispatch(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::consumeSingleRowMoEOverlayActivationDispatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationSingleRowDispatchConsumeLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeSingleRowMoEOverlayActivationDispatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::consumeSingleRowMoEOverlayActivationDispatch] complete one-row packet binding and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationConsumeSingleRowDispatch(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::packSingleRowMoEOverlayActivationReturn(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationSingleRowReturnPackLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "packSingleRowMoEOverlayActivationReturn");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::packSingleRowMoEOverlayActivationReturn] complete one-row packet binding and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationPackSingleRowReturn(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::consumeSingleRowMoEOverlayActivationReturn(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationSingleRowReturnConsumeLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeSingleRowMoEOverlayActivationReturn");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::consumeSingleRowMoEOverlayActivationReturn] complete one-row packet binding and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationConsumeSingleRowReturn(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::packSingleRowMoEOverlayActivationDispatchBatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationSingleRowDispatchBatchLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "packSingleRowMoEOverlayActivationDispatchBatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::packSingleRowMoEOverlayActivationDispatchBatch] persistent topology batch and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationPackSingleRowDispatchBatch(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::consumeSingleRowMoEOverlayActivationReturnBatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationSingleRowReturnBatchLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeSingleRowMoEOverlayActivationReturnBatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::consumeSingleRowMoEOverlayActivationReturnBatch] persistent topology batch, gather scratch, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationConsumeSingleRowReturnBatch(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::consumeMultiRowMoEOverlayActivationReturnBatch(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayActivationMultiRowReturnBatchLaunch &packet)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeMultiRowMoEOverlayActivationReturnBatch");
        if (!stream || !packet.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::consumeMultiRowMoEOverlayActivationReturnBatch] persistent topology batch, row-index scratch, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayActivationConsumeMultiRowReturnBatch(
            &packet, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::publishNodeLocalCanonicalRoutes(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalRoutePublishLaunch &publication)
    {
        void *stream = explicitMoELaunchStream(
            launch, "publishNodeLocalCanonicalRoutes");
        if (!stream || !publication.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::publishNodeLocalCanonicalRoutes] complete mapped lane, route assignment, canonical contribution, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayPublishNodeLocalCanonicalRoutes(
            &publication, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::acquireNodeLocalCanonicalRoutes(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalRouteConsumeLaunch &consumption)
    {
        void *stream = explicitMoELaunchStream(
            launch, "acquireNodeLocalCanonicalRoutes");
        if (!stream || !consumption.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::acquireNodeLocalCanonicalRoutes] persistent peer bindings, route assignment, validation storage, output, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayAcquireNodeLocalCanonicalRoutes(
            &consumption, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::stageNodeLocalCanonicalRoutes(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalRouteConsumeLaunch &consumption)
    {
        void *stream = explicitMoELaunchStream(
            launch, "stageNodeLocalCanonicalRoutes");
        if (!stream || !consumption.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::stageNodeLocalCanonicalRoutes] mapped sources, root scratch, route assignment, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayStageNodeLocalCanonicalRoutes(
            &consumption, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::foldNodeLocalCanonicalRoutes(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalRouteConsumeLaunch &consumption)
    {
        void *stream = explicitMoELaunchStream(
            launch, "foldNodeLocalCanonicalRoutes");
        if (!stream || !consumption.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::foldNodeLocalCanonicalRoutes] persistent peer bindings, staged payload, route assignment, validation storage, output, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayFoldNodeLocalCanonicalRoutes(
            &consumption, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::consumeSingleRowNodeLocalCanonicalRoutes(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalRouteConsumeLaunch &consumption)
    {
        void *stream = explicitMoELaunchStream(
            launch, "consumeSingleRowNodeLocalCanonicalRoutes");
        if (!stream || !consumption.valid() ||
            consumption.physical_rows != 1u ||
            consumption.top_k > 32u || consumption.peer_count > 8u)
        {
            LOG_ERROR("[ROCmMoEKernel::consumeSingleRowNodeLocalCanonicalRoutes] single-row persistent peer bindings, route assignment, validation storage, output, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayConsumeSingleRowNodeLocalCanonicalRoutes(
            &consumption, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::beginNodeLocalDensePublication(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalDensePublicationLaunch &publication)
    {
        void *stream = explicitMoELaunchStream(
            launch, "beginNodeLocalDensePublication");
        if (!stream || !publication.valid() ||
            !publication.binding.isRoot())
        {
            LOG_ERROR("[ROCmMoEKernel::beginNodeLocalDensePublication] root binding, bounded payload, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayBeginNodeLocalDensePublication(
            &publication, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::finishNodeLocalDensePublication(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalDensePublicationLaunch &publication)
    {
        void *stream = explicitMoELaunchStream(
            launch, "finishNodeLocalDensePublication");
        if (!stream || !publication.valid() ||
            !publication.binding.isRoot())
        {
            LOG_ERROR("[ROCmMoEKernel::finishNodeLocalDensePublication] root binding, bounded payload, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayFinishNodeLocalDensePublication(
            &publication, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::beginNodeLocalDensePublicationConsume(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalDensePublicationLaunch &publication)
    {
        void *stream = explicitMoELaunchStream(
            launch, "beginNodeLocalDensePublicationConsume");
        if (!stream || !publication.valid() || publication.binding.isRoot())
        {
            LOG_ERROR("[ROCmMoEKernel::beginNodeLocalDensePublicationConsume] peer binding, bounded payload, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayBeginNodeLocalDensePublicationConsume(
            &publication, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::finishNodeLocalDensePublicationConsume(
        const MoEKernelLaunchContext &launch,
        const MoENodeLocalDensePublicationLaunch &publication)
    {
        void *stream = explicitMoELaunchStream(
            launch, "finishNodeLocalDensePublicationConsume");
        if (!stream || !publication.valid() || publication.binding.isRoot())
        {
            LOG_ERROR("[ROCmMoEKernel::finishNodeLocalDensePublicationConsume] peer binding, bounded payload, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayFinishNodeLocalDensePublicationConsume(
            &publication, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::runMoEOverlayDeviceControllerAction(
        const MoEKernelLaunchContext &launch,
        const MoEOverlayDeviceControllerActionLaunch &action)
    {
        void *stream = explicitMoELaunchStream(
            launch, "runMoEOverlayDeviceControllerAction");
        if (!stream || !action.valid())
        {
            LOG_ERROR("[ROCmMoEKernel::runMoEOverlayDeviceControllerAction] a valid mapped controller action and explicit HIP stream are required");
            return false;
        }
        return hipMoEOverlayRunDeviceControllerAction(
            &action, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::beginMoEOverlayServiceTelemetry(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayServiceTelemetrySample *sample)
    {
        void *stream = explicitMoELaunchStream(
            launch, "beginMoEOverlayServiceTelemetry");
        if (!stream || !sample)
            return false;
        return hipMoEOverlayBeginServiceTelemetry(
            sample, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::finishMoEOverlayServiceTelemetry(
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
        return hipMoEOverlayFinishServiceTelemetry(
            runtime_layer,
            layer_telemetry,
            sample,
            num_experts,
            static_cast<std::uint32_t>(hint),
            runtime_graph_role,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::publishMoEOverlayServiceTelemetry(
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
        return hipMoEOverlayPublishServiceTelemetry(
            telemetry,
            samples,
            layer_count,
            participant_id,
            publication,
            device_ordinal_,
            stream);
    }

    ROCmMoEKernel::ROCmMoEKernel(int device_ordinal)
        : device_ordinal_(device_ordinal)
    {
        auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_ordinal);
        ROCmKernelBase::setDeviceContext(&ctx);

        // Create hipBLAS GEMM kernel using device context (shares hipBLAS handle)
        blas_gemm_ = std::make_unique<rocm::HipBLASGemmKernel>(&ctx);

        /*
         * Construction establishes device resources only. The graph executor
         * has not assigned a producer stream yet, so binding hipBLAS here would
         * either consult an implicit context default or make construction
         * order-dependent. bindGPUStream() performs the parent/child stream
         * transition atomically once the executor supplies its exact stream.
         */
        LOG_TRACE("[ROCmMoEKernel] Created unbound for ROCm device "
                  << device_ordinal);
    }

    void ROCmMoEKernel::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        const uint64_t new_workspace_id = workspace ? workspace->id() : 0;
        if (workspace_ == workspace && bound_workspace_id_ == new_workspace_id)
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
        ROCmKernelBase::bindWorkspace(workspace);
        bound_workspace_id_ = new_workspace_id;
        clearWorkspaceScratchBindings();
        if (workspace_)
            (void)rebindGroupedDescriptorTablesToWorkspace("bindWorkspace");
    }

    bool ROCmMoEKernel::bindWorkspaceBuffer(
        void **ptr,
        const char *name,
        size_t bytes,
        const char *context)
    {
        if (!ptr || !name || bytes == 0)
            return false;
        if (!validateROCmWorkspaceBinding(workspace_, device_ordinal_, "ROCmMoEKernel"))
        {
            LOG_ERROR("[ROCmMoEKernel] " << context
                                         << " requires graph-owned MoE workspace");
            return false;
        }
        void *buffer = workspace_->getBuffer(name);
        const size_t available = workspace_->getBufferSize(name);
        if (!buffer || available < bytes)
        {
            LOG_ERROR("[ROCmMoEKernel] " << context << " missing required workspace buffer '"
                                         << name << "' (need " << bytes << " bytes, have "
                                         << available << ")");
            return false;
        }
        *ptr = buffer;
        scratch_workspace_bound_ = true;
        return true;
    }

    bool ROCmMoEKernel::bindGroupedDescriptorTableSlot(
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
            LOG_ERROR("[ROCmMoEKernel] " << context
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
            LOG_ERROR("[ROCmMoEKernel] " << context
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

    void ROCmMoEKernel::clearWorkspaceScratchBindings() noexcept
    {
        if(original_m1_hidden_q8_read_)
            original_m1_hidden_q8_read_lifecycle_=OriginalM1HiddenQ8ReadLifecycle::Invalidated;
        original_m1_hidden_q8_read_.reset();
        gate_up_last_producer_.clear();
        down_last_producer_.clear();
        paired_router_descriptors_.reset();
        grouped_prefill_grouping_.invalidate();
        d_write_heads_ = nullptr;
        d_staging_indices_ = nullptr;
        d_staging_weights_ = nullptr;
        d_grouped_gate_ptrs_ = nullptr;
        d_grouped_up_ptrs_ = nullptr;
        d_grouped_expert_ids_ = nullptr;
        d_grouped_decode_weights_ = nullptr;
        d_grouped_down_descs_ = nullptr;
        d_grouped_swiglu_int8_ = nullptr;
        d_grouped_swiglu_scales_ = nullptr;
        d_grouped_down_partials_ = nullptr;
        d_grouped_gate_output_ptrs_ = nullptr;
        d_grouped_up_output_ptrs_ = nullptr;
        d_grouped_gateup_expert_ids_ = nullptr;
        d_grouped_hidden_int8_ = nullptr;
        d_grouped_hidden_scales_ = nullptr;
        d_grouped_gateup_gate_partials_ = nullptr;
        d_grouped_gateup_up_partials_ = nullptr;
        d_shared_gate_scratch_ = nullptr;
        d_route_logits_ = nullptr;
        d_route_logits_partials_ = nullptr;
        d_router_q8_hidden_ = nullptr;
        d_router_q8_hidden_scales_ = nullptr;
        invalidateRouterQ8HiddenPublication();
        d_group_int_indices_ = nullptr;
        d_group_offsets_ = nullptr;
        d_group_counts_ = nullptr;
        d_group_max_tokens_ = nullptr;
        d_group_expert_mask_ = nullptr;
        group_expert_mask_workspace_lease_.reset();
        d_group_token_indices_ = nullptr;
        d_group_original_to_grouped_ = nullptr;
        d_group_weights_ = nullptr;
        d_group_active_expert_ids_ = nullptr;
        d_prefill_A_int8_ = nullptr;
        d_prefill_A_scales_ = nullptr;
        d_prefill_swiglu_int8_ = nullptr;
        d_prefill_swiglu_scales_ = nullptr;
        d_prefill_gate_ = nullptr;
        d_prefill_up_ = nullptr;
        d_prefill_work_directory_ = nullptr;
        d_grouped_population_selection_ = nullptr;

        max_write_heads_experts_ = 0;
        staging_capacity_ = 0;
        grouped_decode_active_cap_ = 0;
        grouped_decode_intermediate_cap_ = 0;
        grouped_decode_d_model_cap_ = 0;
        grouped_gateup_active_cap_ = 0;
        grouped_gateup_d_model_cap_ = 0;
        grouped_gateup_intermediate_cap_ = 0;
        shared_gate_scratch_capacity_ = 0;
        route_logits_capacity_ = 0;
        route_logits_partials_capacity_ = 0;
        router_q8_hidden_rows_cap_ = 0;
        router_q8_hidden_d_model_cap_ = 0;
        router_q8_hidden_blocks_cap_ = 0;
        invalidateRouterQ8HiddenPublication();
        group_active_expert_slots_ = 0;
        group_slots_cap_ = 0;
        group_experts_cap_ = 0;
        group_expert_mask_cap_ = 0;
        group_expert_mask_hash_ = 0;
        group_expert_mask_num_experts_ = 0;
        group_expert_mask_active_experts_ = 0;
        group_expert_mask_published_ = false;
        prefill_slots_cap_ = 0;
        prefill_d_model_cap_ = 0;
        prefill_intermediate_cap_ = 0;
        router_q8_gate_cache_.clear();
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
        fixed_gateup_metadata_states_.clear();
        fixed_down_metadata_states_.clear();
        runtime_pointer_workspace_owners_.reset();
        gateup_pointer_slot_ready_.fill(false);
        down_pointer_slot_ready_.fill(false);
        scratch_workspace_bound_ = false;
    }

    bool ROCmMoEKernel::publishGroupedDownDescriptorTable(
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
            LOG_ERROR("[ROCmMoEKernel] "
                      << (context ? context : "down descriptor publication")
                      << " requires a bound workspace and complete host table");
            return false;
        }
        hipStream_t stream = static_cast<hipStream_t>(getStream());
        if (!stream)
        {
            LOG_ERROR("[ROCmMoEKernel] "
                      << (context ? context : "down descriptor publication")
                      << " requires an explicit HIP stream");
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
                    kROCmGroupedDownDescriptorLeaseDomain,
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
                                MoEWorkspaceBuffers::
                                    ROCM_GROUPED_DOWN_DESC_TABLES,
                                slot,
                                table.num_experts,
                                &slot_base,
                                "ROCm grouped floating down descriptor publication"))
                        {
                            return {};
                        }
                        auto *device_descs =
                            reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                                slot_base);

                        hipEvent_t ready_event = nullptr;
                        hipError_t err = hipEventCreateWithFlags(
                            &ready_event,
                            hipEventDisableTiming);
                        if (err != hipSuccess || !ready_event)
                            return {};
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
                                        (void)hipEventDestroy(
                                            static_cast<hipEvent_t>(
                                                value->ready_event));
                                    }
                                    delete value;
                                });
                        err = empty_original_image
                            ? hipMemsetAsync(device_descs,0,desc_bytes,stream)
                            : hipMemcpyAsync(device_descs,table.host_floating_descs.data(),desc_bytes,hipMemcpyHostToDevice,stream);
                        if (err != hipSuccess)
                            return {};
                        err = hipEventRecord(ready_event, stream);
                        if (err != hipSuccess)
                            std::terminate();
                        return publication;
                    });
            if (!publication_result)
                return false;

            auto publication = std::static_pointer_cast<
                GroupedDescriptorWorkspacePublication>(
                publication_result.publication);
            const hipError_t wait_err = hipStreamWaitEvent(
                stream,
                static_cast<hipEvent_t>(publication->ready_event),
                0);
            if (wait_err != hipSuccess)
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
                kROCmGroupedDownDescriptorLeaseDomain,
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
                                ROCM_GROUPED_DOWN_DESC_TABLES,
                            slot,
                            table.num_experts,
                            &device_descs,
                            "ROCm grouped down descriptor publication"))
                    {
                        return {};
                    }

                    hipEvent_t ready_event = nullptr;
                    hipError_t err = hipEventCreateWithFlags(
                        &ready_event,
                        hipEventDisableTiming);
                    if (err != hipSuccess || !ready_event)
                    {
                        LOG_ERROR("[ROCmMoEKernel] Failed to create grouped "
                                  "down descriptor readiness event: "
                                  << hipGetErrorString(err));
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
                                    (void)hipEventDestroy(
                                        static_cast<hipEvent_t>(
                                            value->ready_event));
                                }
                                delete value;
                            });

                    err = empty_original_image
                            ? hipMemsetAsync(device_descs,0,desc_bytes,stream)
                            : hipMemcpyAsync(device_descs,table.host_descs.data(),desc_bytes,hipMemcpyHostToDevice,stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel] Grouped down descriptor "
                                  "H2D publication failed: "
                                  << hipGetErrorString(err));
                        return {};
                    }
                    err = hipEventRecord(ready_event, stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel] Grouped down descriptor "
                                  "publication cannot record readiness: "
                                  << hipGetErrorString(err));
                        std::terminate();
                    }
                    return publication;
                });
        if (!publication_result)
        {
            LOG_ERROR("[ROCmMoEKernel] Failed to publish grouped down "
                      "descriptor table");
            return false;
        }

        auto publication =
            std::static_pointer_cast<
                GroupedDescriptorWorkspacePublication>(
                publication_result.publication);
        const hipError_t wait_err = hipStreamWaitEvent(
            stream,
            static_cast<hipEvent_t>(publication->ready_event),
            0);
        if (wait_err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel] Failed to adopt grouped down "
                      "descriptor publication: "
                      << hipGetErrorString(wait_err));
            return false;
        }

        table.device_descs = static_cast<DeviceNativeQuantisedMatrixDesc *>(
            publication->primary_descs);
        table.device_floating_descs = nullptr;
        table.workspace_publication = std::move(publication);
        table.workspace_slot = publication_result.slot;
        return true;
    }

    bool ROCmMoEKernel::publishGroupedGateUpDescriptorTable(
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
            LOG_ERROR("[ROCmMoEKernel] "
                      << (context ? context : "gate/up descriptor publication")
                      << " requires a bound workspace and complete host tables");
            return false;
        }
        hipStream_t stream = static_cast<hipStream_t>(getStream());
        if (!stream)
        {
            LOG_ERROR("[ROCmMoEKernel] "
                      << (context ? context : "gate/up descriptor publication")
                      << " requires an explicit HIP stream");
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
                    kROCmGroupedGateUpDescriptorLeaseDomain,
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
                                MoEWorkspaceBuffers::
                                    ROCM_GROUPED_GATE_DESC_TABLES,
                                slot,
                                table.num_experts,
                                &gate_slot_base,
                                "ROCm grouped floating gate descriptor publication") ||
                            !bindGroupedDescriptorTableSlot(
                                MoEWorkspaceBuffers::
                                    ROCM_GROUPED_UP_DESC_TABLES,
                                slot,
                                table.num_experts,
                                &up_slot_base,
                                "ROCm grouped floating up descriptor publication"))
                        {
                            return {};
                        }
                        auto *device_gate_descs =
                            reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                                gate_slot_base);
                        auto *device_up_descs =
                            reinterpret_cast<DeviceMoEFloatingMatrixDesc *>(
                                up_slot_base);

                        hipEvent_t ready_event = nullptr;
                        hipError_t err = hipEventCreateWithFlags(
                            &ready_event,
                            hipEventDisableTiming);
                        if (err != hipSuccess || !ready_event)
                            return {};
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
                                        (void)hipEventDestroy(
                                            static_cast<hipEvent_t>(
                                                value->ready_event));
                                    }
                                    delete value;
                                });
                        err = empty_original_image
                            ? hipMemsetAsync(device_gate_descs,0,desc_bytes,stream)
                            : hipMemcpyAsync(device_gate_descs,table.host_floating_gate_descs.data(),desc_bytes,hipMemcpyHostToDevice,stream);
                        if (err != hipSuccess)
                            return {};
                        err = empty_original_image
                            ? hipMemsetAsync(device_up_descs,0,desc_bytes,stream)
                            : hipMemcpyAsync(device_up_descs,table.host_floating_up_descs.data(),desc_bytes,hipMemcpyHostToDevice,stream);
                        if (err != hipSuccess)
                            std::terminate();
                        err = hipEventRecord(ready_event, stream);
                        if (err != hipSuccess)
                            std::terminate();
                        return publication;
                    });
            if (!publication_result)
                return false;

            auto publication = std::static_pointer_cast<
                GroupedDescriptorWorkspacePublication>(
                publication_result.publication);
            const hipError_t wait_err = hipStreamWaitEvent(
                stream,
                static_cast<hipEvent_t>(publication->ready_event),
                0);
            if (wait_err != hipSuccess)
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
                kROCmGroupedGateUpDescriptorLeaseDomain,
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
                                ROCM_GROUPED_GATE_DESC_TABLES,
                            slot,
                            table.num_experts,
                            &device_gate_descs,
                            "ROCm grouped gate descriptor publication") ||
                        !bindGroupedDescriptorTableSlot(
                            MoEWorkspaceBuffers::
                                ROCM_GROUPED_UP_DESC_TABLES,
                            slot,
                            table.num_experts,
                            &device_up_descs,
                            "ROCm grouped up descriptor publication"))
                    {
                        return {};
                    }

                    hipEvent_t ready_event = nullptr;
                    hipError_t err = hipEventCreateWithFlags(
                        &ready_event,
                        hipEventDisableTiming);
                    if (err != hipSuccess || !ready_event)
                    {
                        LOG_ERROR("[ROCmMoEKernel] Failed to create grouped "
                                  "gate/up descriptor readiness event: "
                                  << hipGetErrorString(err));
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
                                    (void)hipEventDestroy(
                                        static_cast<hipEvent_t>(
                                            value->ready_event));
                                }
                                delete value;
                            });

                    err = empty_original_image
                            ? hipMemsetAsync(device_gate_descs,0,desc_bytes,stream)
                            : hipMemcpyAsync(device_gate_descs,table.host_gate_descs.data(),desc_bytes,hipMemcpyHostToDevice,stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel] Grouped gate descriptor "
                                  "H2D publication failed: "
                                  << hipGetErrorString(err));
                        return {};
                    }
                    err = empty_original_image
                            ? hipMemsetAsync(device_up_descs,0,desc_bytes,stream)
                            : hipMemcpyAsync(device_up_descs,table.host_up_descs.data(),desc_bytes,hipMemcpyHostToDevice,stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel] Grouped up descriptor H2D "
                                  "publication failed after gate submission: "
                                  << hipGetErrorString(err));
                        std::terminate();
                    }
                    err = hipEventRecord(ready_event, stream);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel] Grouped gate/up descriptor "
                                  "publication cannot record readiness: "
                                  << hipGetErrorString(err));
                        std::terminate();
                    }
                    return publication;
                });
        if (!publication_result)
        {
            LOG_ERROR("[ROCmMoEKernel] Failed to publish grouped gate/up "
                      "descriptor table");
            return false;
        }

        auto publication =
            std::static_pointer_cast<
                GroupedDescriptorWorkspacePublication>(
                publication_result.publication);
        const hipError_t wait_err = hipStreamWaitEvent(
            stream,
            static_cast<hipEvent_t>(publication->ready_event),
            0);
        if (wait_err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel] Failed to adopt grouped gate/up "
                      "descriptor publication: "
                      << hipGetErrorString(wait_err));
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

    bool ROCmMoEKernel::rebindGroupedDescriptorTablesToWorkspace(const char *context)
    {
        if (grouped_down_desc_tables_.empty() && grouped_gateup_desc_tables_.empty())
            return true;
        if (rejectDecodeStagingDuringCapture("rebind grouped descriptor tables"))
            return false;
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

    ROCmMoEKernel::~ROCmMoEKernel()
    {
        (void)setMoEDevice(device_ordinal_, "destructor");
        clearWorkspaceScratchBindings();
    }

    void ROCmMoEKernel::resetDynamicState()
    {
        grouped_prefill_grouping_.invalidate();
        if (!setMoEDevice(device_ordinal_, "resetDynamicState"))
            return;

        /*
         * This hook is a hard kernel-dynamic reset, not the replay-preserving
         * request boundary. Captured HIP graph replay keeps kernel-owned
         * workspace tables alive by avoiding KernelFactory::resetAllDynamicState().
         * When this hook is invoked, grouped scratch, mask upload hashes,
         * histogram buffers, pointer-table caches, and descriptor registries
         * must be rebuilt before the next eager warmup or capture.
         */
        clearWorkspaceScratchBindings();
        grouped_down_desc_tables_.clear();
        grouped_gateup_desc_tables_.clear();
    }

    bool ROCmMoEKernel::runMoEOverlayEpochParentAction(
        const MoEKernelLaunchContext &launch,DeviceMoEOverlayEpochControl *control,
        DeviceMoEOverlayEpochStatus *status,DeviceMoEOverlayEpochParentAction action)
    {
        void *stream=explicitMoELaunchStream(launch,"runMoEOverlayEpochParentAction");
        if(!stream || !control || !status)return false;
        return hipMoEOverlayEpochParentAction(control,status,action,device_ordinal_,stream);
    }

    bool ROCmMoEKernel::acquireMoEOverlayEpoch(
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
            LOG_ERROR("[ROCmMoEKernel::acquireMoEOverlayEpoch] control, ticket, status, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayEpochAcquire(
            control,
            ticket,
            status,
            external_admission_epoch,
            admission_barrier,
            peer_placement_epoch,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::releaseMoEOverlayEpoch(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        DeviceMoEOverlayEpochTicket *ticket,
        DeviceMoEOverlayEpochStatus *status)
    {
        void *stream = explicitMoELaunchStream(
            launch, "releaseMoEOverlayEpoch");
        if (!stream || !control || !ticket || !status)
        {
            LOG_ERROR("[ROCmMoEKernel::releaseMoEOverlayEpoch] control, ticket, status, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayEpochRelease(
            control, ticket, status, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::reserveMoEOverlayEpochCandidate(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *status)
    {
        void *stream = explicitMoELaunchStream(
            launch, "reserveMoEOverlayEpochCandidate");
        if (!stream || !control || !candidate_epoch || !status)
        {
            LOG_ERROR("[ROCmMoEKernel::reserveMoEOverlayEpochCandidate] control, device epoch, status, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayEpochReserveCandidate(
            control, candidate_epoch, status, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::markMoEOverlayEpochCandidateReady(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *status)
    {
        void *stream = explicitMoELaunchStream(
            launch, "markMoEOverlayEpochCandidateReady");
        if (!stream || !control || !candidate_epoch || !status)
        {
            LOG_ERROR("[ROCmMoEKernel::markMoEOverlayEpochCandidateReady] control, device epoch, status, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayEpochMarkCandidateReady(
            control, candidate_epoch, status, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::publishMoEOverlayEpochCandidate(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *status)
    {
        void *stream = explicitMoELaunchStream(
            launch, "publishMoEOverlayEpochCandidate");
        if (!stream || !control || !candidate_epoch || !status)
        {
            LOG_ERROR("[ROCmMoEKernel::publishMoEOverlayEpochCandidate] control, device epoch, status, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayEpochPublishCandidate(
            control, candidate_epoch, status, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::abortMoEOverlayEpochCandidate(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        DeviceMoEOverlayEpochStatus *status)
    {
        void *stream = explicitMoELaunchStream(
            launch, "abortMoEOverlayEpochCandidate");
        if (!stream || !control || !candidate_epoch || !status)
        {
            LOG_ERROR("[ROCmMoEKernel::abortMoEOverlayEpochCandidate] control, device epoch, status, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayEpochAbortCandidate(
            control, candidate_epoch, status, device_ordinal_, stream);
    }

    bool ROCmMoEKernel::retireMoEOverlayEpoch(
        const MoEKernelLaunchContext &launch,
        DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *retiring_epoch,
        DeviceMoEOverlayEpochStatus *status)
    {
        void *stream = explicitMoELaunchStream(
            launch, "retireMoEOverlayEpoch");
        if (!stream || !control || !retiring_epoch || !status)
        {
            LOG_ERROR("[ROCmMoEKernel::retireMoEOverlayEpoch] control, device epoch, status, and explicit stream are required");
            return false;
        }
        return hipMoEOverlayEpochRetire(
            control, retiring_epoch, status, device_ordinal_, stream);
    }

    void ROCmMoEKernel::syncBlasStream()
    {
        if (blas_gemm_)
            blas_gemm_->bindStream(ExplicitGPUStream{ROCmKernelBase::getStream()});
    }

    void ROCmMoEKernel::clearGPUStreamBinding()
    {
        ROCmKernelBase::clearGPUStreamBinding();
        if (blas_gemm_)
            blas_gemm_->clearStreamBinding();
    }

    bool ROCmMoEKernel::ensureSharedGateScratchCapacity(int seq_len)
    {
        if (seq_len <= shared_gate_scratch_capacity_)
            return true;

        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_shared_gate_scratch_),
                                 MoEWorkspaceBuffers::ROCM_SHARED_GATE,
                                 static_cast<size_t>(seq_len) * sizeof(float),
                                 "ensureSharedGateScratchCapacity"))
        {
            shared_gate_scratch_capacity_ = 0;
            return false;
        }
        shared_gate_scratch_capacity_ = seq_len;
        return true;
    }

    bool ROCmMoEKernel::ensureRouteBufferCapacity(size_t logits_count)
    {
        if (logits_count > route_logits_capacity_)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_route_logits_),
                                     MoEWorkspaceBuffers::ROUTE_LOGITS,
                                     logits_count * sizeof(float),
                                     "ensureRouteBufferCapacity(logits)"))
            {
                route_logits_capacity_ = 0;
                return false;
            }
            route_logits_capacity_ = logits_count;
        }

        return d_route_logits_ != nullptr;
    }

    bool ROCmMoEKernel::ensureRouteLogitsPartialsCapacity(size_t partial_count)
    {
        if (partial_count == 0)
            return false;

        if (partial_count <= route_logits_partials_capacity_)
            return d_route_logits_partials_ != nullptr;

        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_route_logits_partials_),
                                 MoEWorkspaceBuffers::ROCM_ROUTE_LOGITS_PARTIALS,
                                 partial_count * sizeof(float),
                                 "ensureRouteLogitsPartialsCapacity"))
        {
            d_route_logits_partials_ = nullptr;
            route_logits_partials_capacity_ = 0;
            return false;
        }

        route_logits_partials_capacity_ = partial_count;
        return true;
    }

    bool ROCmMoEKernel::ensureRouterQ8HiddenScratchCapacity(int rows, int d_model)
    {
        if (rows <= 0 || d_model <= 0 || (d_model % 32) != 0)
            return false;

        const int blocks_per_row = d_model / 32;
        if (rows <= router_q8_hidden_rows_cap_ &&
            d_model <= router_q8_hidden_d_model_cap_ &&
            blocks_per_row <= router_q8_hidden_blocks_cap_ &&
            d_router_q8_hidden_ && d_router_q8_hidden_scales_)
        {
            return true;
        }

        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_router_q8_hidden_),
                                 MoEWorkspaceBuffers::ROCM_ROUTER_Q8_HIDDEN,
                                 static_cast<size_t>(rows) *
                                     static_cast<size_t>(d_model) * sizeof(int8_t),
                                 "ensureRouterQ8HiddenScratchCapacity(hidden)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_router_q8_hidden_scales_),
                                 MoEWorkspaceBuffers::ROCM_ROUTER_Q8_SCALES,
                                 static_cast<size_t>(rows) *
                                     static_cast<size_t>(blocks_per_row) * sizeof(float),
                                 "ensureRouterQ8HiddenScratchCapacity(scales)"))
        {
            d_router_q8_hidden_ = nullptr;
            d_router_q8_hidden_scales_ = nullptr;
            router_q8_hidden_rows_cap_ = 0;
            router_q8_hidden_d_model_cap_ = 0;
            router_q8_hidden_blocks_cap_ = 0;
            return false;
        }

        router_q8_hidden_rows_cap_ = rows;
        router_q8_hidden_d_model_cap_ = d_model;
        router_q8_hidden_blocks_cap_ = blocks_per_row;
        return true;
    }

    bool ROCmMoEKernel::prepareOriginalM1HiddenQ8Publication(const TensorBase* hidden,DeviceRowRange rows)
    {
        const auto identity=rows.captureIdentity();auto& publication=router_q8_hidden_publication_;
        if(isGraphCaptureActive()||!publication||!publication->requiredOriginalM1Source()||
           hidden!=publication->requiredOriginalM1Source()||identity!=DeviceRowRange::fullyActive(1).captureIdentity()||!workspace_)
            throw std::invalid_argument("Original M1 hidden Q8 requires its cold declared fully-active source");
        const auto stream=getStream();const int width=publication->requiredOriginalM1Width();
        if(router_q8_publication_access_!=MoERouterQ8PublicationAccess::RequiredConsumer) {
            if(!publication->original_m1_binding)
                publication->original_m1_binding=std::make_shared<OriginalM1HiddenQ8Binding>(
                    DeviceId::rocm(device_ordinal_),*hidden,width,stream,*workspace_,OriginalM1HiddenQ8Bank::RouterROCm);
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

    std::uint64_t ROCmMoEKernel::originalM1HiddenQ8BindingSignature() const
    {
        if(!router_q8_hidden_publication_||!router_q8_hidden_publication_->requiredOriginalM1Source())return 0;
        const auto& publication=*router_q8_hidden_publication_;
        if(publication.original_m1_binding) {
            const auto stream=ExplicitGPUStream(gpu_stream_).get();
            if(original_m1_hidden_q8_read_lifecycle_==OriginalM1HiddenQ8ReadLifecycle::Invalidated)
                throw std::logic_error("Original M1 hidden Q8 reader lost its prepared scratch lifetime");
            if(router_q8_publication_access_!=MoERouterQ8PublicationAccess::RequiredConsumer)
                publication.original_m1_binding->validate(*publication.requiredOriginalM1Source(),stream,workspace_);
            else if(original_m1_hidden_q8_read_)
                original_m1_hidden_q8_read_->validate(*publication.requiredOriginalM1Source(),stream,workspace_);
        }
        // Declaration identity stays stable before/after first cold preparation;
        // leases independently authenticate every embedded pointer on replay.
        return 0x4d31513800000001ULL;
    }

    bool ROCmMoEKernel::prepareRequiredRouterQ8Hidden(int rows, int width, int top_k)
    {
        (void)top_k;
        if (!router_q8_hidden_publication_ ||
            !router_q8_hidden_publication_->hasRequiredConsumer())
            return true;
        if (router_q8_publication_access_ == MoERouterQ8PublicationAccess::RequiredConsumer ||
            !router_q8_hidden_publication_->acceptsProducerRows(rows, width))
        {
            LOG_ERROR("[ROCmMoEKernel] required Q8 producer geometry/access mismatch");
            return false;
        }
        return ensureRouterQ8HiddenScratchCapacity(rows, width);
    }

    bool ROCmMoEKernel::publishRequiredRouterQ8Hidden(
        const float *source, int rows, int width, int top_k, const int *live_rows)
    {
        if (!router_q8_hidden_publication_ ||
            !router_q8_hidden_publication_->hasRequiredConsumer())
            return true;
        if (!prepareRequiredRouterQ8Hidden(rows, width, top_k)) return false;
        const bool affine=router_q8_publication_access_==MoERouterQ8PublicationAccess::ProducerAndConsumerWithExactK32Sums;
        if (affine) {
            namespace frontier=llaminar2::rocm::grouped_ffn;
            auto* raw=workspace_?workspace_->getBuffer(MoEWorkspaceBuffers::PREFILL_A_INT8):nullptr;
            if(!raw || rows!=frontier::capacity || width!=frontier::hidden || top_k!=frontier::topK || !live_rows ||
               workspace_->getBufferSize(MoEWorkspaceBuffers::PREFILL_A_INT8)<frontier::k32SumWorkspaceBytes)
                throw std::logic_error("Exact K32 sum router lost its admitted counted workspace");
            if(!hipMoE_publish_original_hidden_q8_with_k32_sums(source,d_router_q8_hidden_,
                d_router_q8_hidden_scales_,frontier::exactK32Sums(raw)->hiddenRows,device_ordinal_,getStream(),live_rows))
                return false;
        } else if(!hipMoE_publish_original_hidden_q8_rows(source, d_router_q8_hidden_,
                d_router_q8_hidden_scales_, rows, width, device_ordinal_, getStream(), live_rows))
            return false;
        publishRouterQ8Hidden(source, rows,
            isGraphCaptureActive() ||
                (deviceContext() && deviceContext()->isDeviceGraphCaptureActive()) ||
                isHipStreamCapturing(getStream()),
            RouterQ8Basis::Original);
        return true;
    }

    void ROCmMoEKernel::invalidateRouterQ8HiddenPublication() noexcept
    {
        if (router_q8_publication_access_ !=
                MoERouterQ8PublicationAccess::RequiredConsumer &&
            router_q8_hidden_publication_)
        {
            router_q8_hidden_publication_->clearPayload();
        }
    }

    bool ROCmMoEKernel::bindRouterQ8HiddenPublication(
        std::shared_ptr<MoERouterQ8HiddenPublication> publication,
        MoERouterQ8PublicationAccess access)
    {
        if (!publication)
        {
            LOG_ERROR("[ROCmMoEKernel::bindRouterQ8HiddenPublication] null publication");
            return false;
        }
        if (publication->device_ordinal >= 0 &&
            (publication->backend != DeviceType::ROCm ||
             publication->device_ordinal != device_ordinal_))
        {
            LOG_ERROR("[ROCmMoEKernel::bindRouterQ8HiddenPublication] device mismatch"
                      << " publication_backend="
                      << static_cast<int>(publication->backend)
                      << " publication_ordinal=" << publication->device_ordinal
                      << " kernel_ordinal=" << device_ordinal_);
            return false;
        }

        publication->backend = DeviceType::ROCm;
        publication->device_ordinal = device_ordinal_;
        if(router_q8_hidden_publication_&&router_q8_hidden_publication_->original_m1_binding&&
           (publication!=router_q8_hidden_publication_||access!=router_q8_publication_access_))
            throw std::logic_error("Captured original M1 hidden Q8 publication/access cannot be rebound");
        // The original routing stage reaffirms producer access on the same
        // shared owner. It cannot erase an exact graph-local demand.
        if(router_q8_publication_access_==MoERouterQ8PublicationAccess::ProducerAndConsumerWithExactK32Sums) {
            if(router_q8_hidden_publication_!=publication || access==MoERouterQ8PublicationAccess::RequiredConsumer)
                throw std::logic_error("Exact K32 sum publication cannot rebind its router owner");
            return true;
        }
        router_q8_hidden_publication_ = std::move(publication);
        router_q8_publication_access_ = access;
        return true;
    }

    void ROCmMoEKernel::publishRouterQ8Hidden(
        const float *source,
        int rows,
        bool recorded_during_capture,
        RouterQ8Basis router_basis) noexcept
    {
        if (router_q8_publication_access_ ==
                MoERouterQ8PublicationAccess::RequiredConsumer ||
            !router_q8_hidden_publication_)
        {
            LOG_ERROR("[ROCmMoEKernel::publishRouterQ8Hidden] consumer-only kernel attempted publication");
            return;
        }
        if (!source || rows <= 0 ||
            rows > router_q8_hidden_rows_cap_ ||
            !d_router_q8_hidden_ || !d_router_q8_hidden_scales_)
        {
            invalidateRouterQ8HiddenPublication();
            return;
        }

        auto &publication = *router_q8_hidden_publication_;
        publication.backend = DeviceType::ROCm;
        publication.device_ordinal = device_ordinal_;
        publication.source_rows = source;
        publication.quantized_rows = d_router_q8_hidden_;
        publication.row_scales = d_router_q8_hidden_scales_;
        publication.published_rows = rows;
        publication.d_model_capacity = router_q8_hidden_d_model_cap_;
        publication.blocks_per_row_capacity =
            router_q8_hidden_blocks_cap_;
        publication.capture_recorded = recorded_during_capture;
        if (router_basis == RouterQ8Basis::SignedHadamard32)
            PerfStatsCollector::addCounter("kernel", "rocm_moe_router_hadamard32_calls", 1.0,
                "router", "rocm:" + std::to_string(device_ordinal_),
                {{"rows", std::to_string(rows)}, {"expert_hidden_basis", "original"}, {"block", "32"}});

    }

    bool ROCmMoEKernel::canReuseRouterQ8Hidden(
        const float *source,
        int rows,
        int d_model) const
    {
        if(router_q8_hidden_publication_&&router_q8_hidden_publication_->requiredOriginalM1Source()) {
            const auto& p=*router_q8_hidden_publication_;const auto& binding=p.original_m1_binding;
            if(!binding||rows!=1||d_model!=p.requiredOriginalM1Width()||source!=binding->source())
                throw std::logic_error("Required original M1 hidden Q8 source/row geometry is absent");
            (void)originalM1HiddenQ8BindingSignature();
            if(p.backend!=DeviceType::ROCm||p.device_ordinal!=device_ordinal_||p.d_model_capacity<d_model||p.blocks_per_row_capacity<d_model/32||
               p.source_rows!=source||p.quantized_rows!=binding->codes()||p.row_scales!=binding->scales()||p.published_rows!=1||
               (p.capture_recorded&&!isGraphCaptureActive()))
                throw std::logic_error("Required original M1 hidden Q8 payload/provenance is absent");
            return true;
        }
        if (!debugEnv().rocm.moe_reuse_router_q8_hidden ||
            !router_q8_hidden_publication_ ||
            rows <= 0 ||
            d_model <= 0 ||
            router_q8_hidden_publication_->backend != DeviceType::ROCm ||
            router_q8_hidden_publication_->device_ordinal != device_ordinal_ ||
            router_q8_hidden_publication_->source_rows != source ||
            router_q8_hidden_publication_->published_rows < rows ||
            router_q8_hidden_publication_->d_model_capacity < d_model ||
            router_q8_hidden_publication_->blocks_per_row_capacity <
                ((d_model + 31) / 32) ||
            !router_q8_hidden_publication_->quantized_rows ||
            !router_q8_hidden_publication_->row_scales)
        {
            return false;
        }

        /*
         * A capture producer and consumer are ordered correctly on the same
         * HIP stream even though neither has run yet.  Outside capture, those
         * recorded kernels have not materialized bytes, so only an eager
         * publication may be reused by eager expert decode.
         */
        return !router_q8_hidden_publication_->capture_recorded ||
               isDecodeGraphCaptureActive();
    }

    const ROCmMoEKernel::RouterQ8GateCacheEntry *ROCmMoEKernel::getOrCreateQ8RouterGateCache(
        const float *gate_device_ptr,
        int d_model,
        int num_experts)
    {
        if (!debugEnv().rocm.moe_router_q8)
            return nullptr;
        if (!gate_device_ptr || d_model <= 0 || num_experts <= 0 ||
            (d_model % 32) != 0)
        {
            LOG_ERROR("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] invalid Q8 router gate request "
                      "(gate_device_ptr=" << static_cast<const void *>(gate_device_ptr)
                      << " d_model=" << d_model
                      << " num_experts=" << num_experts << ")");
            return nullptr;
        }
        if (!workspace_)
        {
            LOG_ERROR("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] "
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

        const bool capture_active = isGraphCaptureActive() ||
                                    (deviceContext() && deviceContext()->isDeviceGraphCaptureActive());
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
        void *stream = getStream();
        if (stream)
        {
            const hipError_t capture_err = hipStreamIsCapturing(static_cast<hipStream_t>(stream), &capture_status);
            if (capture_err != hipSuccess)
                capture_status = hipStreamCaptureStatusNone;
        }
        if (capture_active || capture_status == hipStreamCaptureStatusActive)
        {
            LOG_ERROR("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] Q8 router cache miss during graph capture");
            return nullptr;
        }
        if (!stream)
        {
            LOG_ERROR("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] explicit HIP stream is required");
            return nullptr;
        }

        const size_t d_model_sz = static_cast<size_t>(d_model);
        const size_t experts_sz = static_cast<size_t>(num_experts);
        if (d_model_sz > std::numeric_limits<size_t>::max() / experts_sz)
        {
            LOG_ERROR("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] router gate size overflow");
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
                                 MoEWorkspaceBuffers::ROCM_ROUTER_Q8_GATE_WEIGHTS,
                                 weights_payload_bytes,
                                 "ROCm Q8 router gate weights") ||
            !bindWorkspaceBuffer(&gate_scales_base,
                                 MoEWorkspaceBuffers::ROCM_ROUTER_Q8_GATE_SCALES,
                                 scales_payload_bytes,
                                 "ROCm Q8 router gate scales"))
        {
            return nullptr;
        }

        /*
         * Model router weights are immutable, while the paired routed pipeline
         * kernels are graph-local by design. Publish one conversion through the
         * workspace manager and let every graph adopt it. The setup factory
         * records an event before exposing the publication, so cross-stream
         * adoption is an explicit HIP dependency rather than a host sync.
         */
        const auto publication_result =
            workspace_->getOrCreatePersistentPublication(
                kROCmRouterQ8GatePublicationDomain,
                routerPublicationKey(cache_key),
                MoEWorkspaceBuffers::kRouterGateCacheSlots,
                [&](std::size_t slot) -> std::shared_ptr<void>
                {
                    void *weights = workspace_->getPersistentSlotBuffer(
                        MoEWorkspaceBuffers::ROCM_ROUTER_Q8_GATE_WEIGHTS,
                        MoEWorkspaceBuffers::kRouterGateCacheSlots,
                        slot,
                        weights_payload_bytes);
                    void *scales = workspace_->getPersistentSlotBuffer(
                        MoEWorkspaceBuffers::ROCM_ROUTER_Q8_GATE_SCALES,
                        MoEWorkspaceBuffers::kRouterGateCacheSlots,
                        slot,
                        scales_payload_bytes);
                    if (!weights || !scales)
                        return {};

                    hipEvent_t ready_event = nullptr;
                    hipError_t event_err = hipEventCreateWithFlags(
                        &ready_event, hipEventDisableTiming);
                    if (event_err != hipSuccess || !ready_event)
                    {
                        LOG_ERROR("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] "
                                  "failed to create immutable publication event: "
                                  << hipGetErrorString(event_err));
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
                                    (void)hipEventDestroy(
                                        static_cast<hipEvent_t>(
                                            value->ready_event));
                                }
                                delete value;
                            });

                    if (!hipMoE_quantize_router_gate_q8(
                            gate_device_ptr,
                            static_cast<int8_t *>(weights),
                            static_cast<float *>(scales),
                            d_model,
                            num_experts,
                            device_ordinal_,
                            stream, routerQ8Basis(debugEnv().gemm.moe_router_arithmetic)))
                    {
                        LOG_ERROR("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] "
                                  "FP32->Q8 router gate conversion launch failed");
                        return {};
                    }

                    event_err = hipEventRecord(
                        ready_event, static_cast<hipStream_t>(stream));
                    if (event_err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] "
                                  "failed to record immutable publication event: "
                                  << hipGetErrorString(event_err));
                        /*
                         * Conversion work is already in flight. There is no
                         * coherent unwind without its readiness event, and a
                         * stream/device sync would be an implicit recovery
                         * path. Stop the process before stale bytes can escape.
                         */
                        std::terminate();
                    }
                    return publication;
                });
        if (!publication_result)
        {
            LOG_ERROR("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] "
                      "failed to publish immutable router gate");
            return nullptr;
        }

        auto publication =
            std::static_pointer_cast<RouterGateWorkspacePublication>(
                publication_result.publication);
        const hipError_t wait_err = hipStreamWaitEvent(
            static_cast<hipStream_t>(stream),
            static_cast<hipEvent_t>(publication->ready_event),
            0);
        if (wait_err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel::getOrCreateQ8RouterGateCache] "
                      "failed to adopt immutable router publication: "
                      << hipGetErrorString(wait_err));
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

        LOG_TRACE("[ROCmMoEKernel] Adopted Q8 router gate source_device="
                  << static_cast<const void *>(gate_device_ptr)
                  << " workspace_id=" << cache_key.workspace_id
                  << " slot=" << publication_result.slot
                  << " created=" << publication_result.created
                  << " shape=[" << num_experts << "," << d_model << "] payload_bytes="
                  << element_count << " scale_bytes=" << (scale_count * sizeof(float)));
        return &router_q8_gate_cache_.back();
    }


    // =========================================================================
    // routeCore() — Shared GPU routing logic: gate GEMM + direct FP32 publication.
    // =========================================================================

    bool ROCmMoEKernel::routeCore(
        const float *hidden, const void *gate_weights, TensorType gate_type,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights,
        float *output_indices, float *output_weights,
        const int *device_effective_seq_len,
        const MoERouterOwnedRowOutputs *owned)
    {
        if (!owned && (!output_indices || !output_weights))
        {
            LOG_ERROR("[ROCmMoEKernel::routeCore] final FP32 route outputs are required");
            return false;
        }
        const size_t logits_count = static_cast<size_t>(seq_len) * num_experts;
        if (!ensureRouteBufferCapacity(logits_count))
            return false;
        if (device_effective_seq_len &&
            !validateDevicePointerOrLog(device_effective_seq_len,
                                        device_ordinal_,
                                        "effective prefill sequence length",
                                        "ROCmMoEKernel::routeCore"))
        {
            return false;
        }

        // The two destinations deliberately share gate preparation and hidden-Q8
        // ownership. The packet specialization also performs top-k publication,
        // so complete-route top-k below must not run a second time.
        const auto route_owned = [&](MoERouterPreparedFormat format, const void *gate,
                                     const float *scales = nullptr) {
            const MoERouterOwnedRowsLaunch launch{
                .partition = owned->partition,
                .capacity = seq_len, .width = d_model, .experts = num_experts,
                .top_k = top_k, .normalize = normalize_weights,
                .live_rows = device_effective_seq_len, .hidden = hidden,
                .format = format,
                .q8_basis = format == MoERouterPreparedFormat::BlockQ8 ?
                    routerQ8Basis(debugEnv().gemm.moe_router_arithmetic) : RouterQ8Basis::Original,
                .gate = gate, .gate_scales = scales,
                .hidden_q8 = d_router_q8_hidden_, .hidden_scales = d_router_q8_hidden_scales_,
                .logits = d_route_logits_, .selected = owned->selected,
                .selected_bytes = owned->selected_bytes};
            return rocm::routeOwnedRows(launch, getStream());
        };
        const bool decode_single_token = (seq_len == 1);
        if (owned && decode_single_token) return false;
        bool used_q8_grouped_router = false;
        bool used_fp16_grouped_router = false;
        invalidateRouterQ8HiddenPublication();
        if (!prepareRequiredRouterQ8Hidden(seq_len, d_model, top_k))
            return false;
        const bool router_publishes_q8 = !decode_single_token &&
            gate_type == TensorType::FP32 && debugEnv().rocm.moe_router_q8;
        if (!router_publishes_q8 && !publishRequiredRouterQ8Hidden(
                hidden, seq_len, d_model, top_k, device_effective_seq_len))
            return false;
        if (decode_single_token)
        {
            if (!launchDecodeGateLogitsForGateType(
                    hidden, gate_weights, gate_type, d_route_logits_,
                    d_model, num_experts,
                    device_ordinal_, getStream(),
                    "ROCmMoEKernel::routeCore"))
            {
                return false;
            }
        }
        else
        {
            /*
             * Prefill routing is part of autoregressive state construction, so
             * changing the prefill bucket must not change a row's logits or
             * top-k weights.  Every native gate format therefore uses the same
             * expert-owned, four-row-tiled reduction shape as serial decode.
             * The tile shares weight loads across rows without introducing a
             * batch-shaped GEMM reduction or floating-point atomic publication.
             */
            bool logits_ready = false;
            if (gate_type == TensorType::FP32)
            {
                const auto &rocm_env = debugEnv().rocm;
                if (rocm_env.moe_router_q8)
                {
                    if ((d_model % 32) != 0 ||
                        !ensureRouterQ8HiddenScratchCapacity(seq_len, d_model))
                    {
                        LOG_ERROR("[ROCmMoEKernel::routeCore] batch-invariant Q8 prefill router "
                                  "requires 32-column blocks and declared row scratch");
                        return false;
                    }
                    const auto *q8_gate = getOrCreateQ8RouterGateCache(
                        static_cast<const float *>(gate_weights),
                        d_model,
                        num_experts);
                    if (!q8_gate)
                    {
                        LOG_ERROR("[ROCmMoEKernel::routeCore] Q8 prefill router gate cache unavailable");
                        return false;
                    }
                    logits_ready = owned
                        ? route_owned(MoERouterPreparedFormat::BlockQ8,
                                      q8_gate->d_gate_weights_q8, q8_gate->d_gate_scales)
                        : hipMoE_gate_logits_q8_weights_decode_equivalent_rows(
                        hidden,
                        d_router_q8_hidden_,
                        d_router_q8_hidden_scales_,
                        q8_gate->d_gate_weights_q8,
                        q8_gate->d_gate_scales,
                        d_route_logits_,
                        seq_len,
                        d_model,
                        num_experts,
                        device_ordinal_,
                        getStream(),
                        device_effective_seq_len, routerQ8Basis(debugEnv().gemm.moe_router_arithmetic));
                    if (logits_ready)
                    {
                        used_q8_grouped_router = true;
                        publishRouterQ8Hidden(
                            hidden,
                            seq_len,
                            isGraphCaptureActive() ||
                                (deviceContext() && deviceContext()->isDeviceGraphCaptureActive()) ||
                                isHipStreamCapturing(getStream()),
                            routerQ8Basis(debugEnv().gemm.moe_router_arithmetic));
                    }
                }
                else
                {
                    logits_ready = owned ? route_owned(MoERouterPreparedFormat::FP32, gate_weights)
                        : launchDecodeEquivalentFP32Rows(
                        hidden,
                        static_cast<const float *>(gate_weights),
                        d_route_logits_,
                        seq_len,
                        d_model,
                        num_experts,
                        device_ordinal_,
                        getStream(),
                        device_effective_seq_len);
                }
            }
            else if (gate_type == TensorType::FP16)
            {
                logits_ready = owned ? route_owned(MoERouterPreparedFormat::FP16, gate_weights)
                    : hipMoE_gate_logits_fp16_decode_equivalent_rows(
                    hidden,
                    gate_weights,
                    d_route_logits_,
                    seq_len,
                    d_model,
                    num_experts,
                    device_ordinal_,
                    getStream(),
                    device_effective_seq_len);
                used_fp16_grouped_router = logits_ready;
            }
            else if (gate_type == TensorType::BF16)
            {
                logits_ready = owned ? route_owned(MoERouterPreparedFormat::BF16, gate_weights)
                    : hipMoE_gate_logits_bf16_decode_equivalent_rows(
                    hidden,
                    gate_weights,
                    d_route_logits_,
                    seq_len,
                    d_model,
                    num_experts,
                    device_ordinal_,
                    getStream(),
                    device_effective_seq_len);
            }
            else
            {
                LOG_ERROR("[ROCmMoEKernel::routeCore] unsupported batch-invariant prefill router gate type "
                          << tensorTypeName(gate_type));
                return false;
            }

            if (!logits_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::routeCore] batch-invariant grouped prefill logits failed");
                return false;
            }
        }

        // The row-owned top-k kernel mirrors serial decode and independently
        // handles every padded graph row from the device effective-length scalar.
        if (!owned && !hipMoE_softmax_topk_decode_equivalent_rows(
                d_route_logits_,
                output_indices,
                output_weights,
                seq_len,
                num_experts,
                top_k,
                normalize_weights,
                device_ordinal_,
                getStream(),
                device_effective_seq_len))
        {
            return false;
        }

        if (!decode_single_token)
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "rocm_moe_batch_invariant_prefill_router_calls",
                1.0,
                "moe",
                DeviceId::rocm(device_ordinal_).to_string(),
                {{"seq_len", std::to_string(seq_len)},
                 {"row_tile", "16"},
                 {"route", used_q8_grouped_router
                               ? "grouped_q8"
                               : (used_fp16_grouped_router ? "grouped_fp16"
                                                          : "grouped_native")},
                 {"device_effective_len", device_effective_seq_len ? "1" : "0"}});
        }

        return true;
    }

    // =========================================================================
    // gatherTokenBatch() — All pointers are device pointers
    // =========================================================================

    void ROCmMoEKernel::gatherTokenBatch(
        const float *hidden,
        float *batch_buffer,
        const int *token_indices,
        int num_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_GATHER, static_cast<hipStream_t>(getStream()));

        if (num_tokens <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "gatherTokenBatch"))
            return;

        if (!hipMoE_gather_tokens(hidden, batch_buffer, token_indices,
                                  num_tokens, d_model,
                                  device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::gatherTokenBatch] kernel launch failed");
        }
    }

    // =========================================================================
    // scatterAddWeighted() — All pointers are device pointers
    // =========================================================================

    void ROCmMoEKernel::scatterAddWeighted(
        float *output,
        const float *expert_output,
        const int *token_indices,
        const float *weights,
        int num_tokens, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SCATTER, static_cast<hipStream_t>(getStream()));

        if (num_tokens <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "scatterAddWeighted"))
            return;

        if (!hipMoE_scatter_add(output, expert_output, token_indices, weights,
                                num_tokens, d_model,
                                device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::scatterAddWeighted] kernel launch failed");
        }
    }

    // =========================================================================
    // sharedExpertGate() — All pointers are device pointers
    //
    // Needs a small scratch buffer for gate values [seq_len floats].
    // Allocates on demand (small — at most a few KB).
    // =========================================================================

    void ROCmMoEKernel::sharedExpertGate(
        const float *input,
        const float *gate_inp,
        float *shared_output,
        int seq_len, int d_model)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SHARED_GATE, static_cast<hipStream_t>(getStream()));

        if (seq_len <= 0 || d_model <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "sharedExpertGate"))
            return;

        if (seq_len == 1)
        {
            if (!hipMoE_shared_expert_gate_decode_fused(input, gate_inp, shared_output,
                                                        d_model,
                                                        device_ordinal_, getStream()))
            {
                LOG_ERROR("[ROCmMoEKernel::sharedExpertGate] fused decode kernel launch failed");
            }
            return;
        }

        if (!ensureSharedGateScratchCapacity(seq_len))
            return;

        if (!hipMoE_shared_expert_gate(input, gate_inp, shared_output, d_shared_gate_scratch_,
                                       seq_len, d_model,
                                       device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::sharedExpertGate] kernel launch failed");
        }
    }

    // =========================================================================
    // swiGLU() — All pointers are device pointers
    // =========================================================================

    void ROCmMoEKernel::swiGLU(float *gate, const float *up, int count)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SWIGLU, static_cast<hipStream_t>(getStream()));

        if (count <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "swiGLU"))
            return;

        if (!hipMoE_swiglu(gate, up, count, device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::swiGLU] kernel launch failed");
        }
    }

    void ROCmMoEKernel::weightedAdd(float *output, const float *input,
                                    float weight, int count)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_SCATTER, static_cast<hipStream_t>(getStream()));

        if (count <= 0)
            return;

        if (!setMoEDevice(device_ordinal_, "weightedAdd"))
            return;

        if (!hipMoE_weighted_add(output, input, weight, count, device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::weightedAdd] kernel launch failed");
        }
    }

    // =========================================================================
    // Device-side token grouping
    // =========================================================================

    /**
     * @brief Reset all grouping publications before the retained HIP pipeline.
     * @param d_routing_indices Original expert IDs, including invalid slots.
     * @param d_routing_weights Immutable per-route probability bits.
     * @param seq_len Admitted token extent, including inactive bucket capacity.
     * @param num_experts Number of expert counters and scatter cursors.
     * @param top_k Route slots per token.
     * @param d_expert_offsets Output exclusive expert-count prefix.
     * @param d_expert_counts Output integer expert counts.
     * @param d_grouped_token_indices Output compact route IDs; unused slots are zero.
     * @param d_grouped_weights Output compact probabilities; unused slots are zero.
     * @return False on binding, initialization, counting, scanning or scatter failure.
     */
    bool ROCmMoEKernel::groupTokensByExpertDevice(
        const int *d_routing_indices,
        const float *d_routing_weights,
        int seq_len, int num_experts, int top_k,
        int *d_expert_offsets,
        int *d_expert_counts,
        int *d_grouped_token_indices,
        float *d_grouped_weights)
    {
        grouped_prefill_grouping_.invalidate();
        return enqueueExpertGrouping(d_routing_indices, d_routing_weights, seq_len, num_experts, top_k,
            d_expert_offsets, d_expert_counts, d_grouped_token_indices, d_grouped_weights, nullptr);
    }

    bool ROCmMoEKernel::enqueueExpertGrouping(
        const int *d_routing_indices, const float *d_routing_weights,
        int seq_len, int num_experts, int top_k, int *d_expert_offsets, int *d_expert_counts,
        int *d_grouped_token_indices, float *d_grouped_weights, const MoEGroupingPackedDirectory *packed)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (seq_len <= 0 || num_experts <= 0 || top_k <= 0)
            return false;

        if (!setMoEDevice(device_ordinal_, "groupTokensByExpertDevice"))
            return false;

        const int total_slots = seq_len * top_k;
        hipStream_t stream = static_cast<hipStream_t>(getStream());
        // Bind the existing arena cursor before the common reset. Counting and
        // scanning do not touch cursors, so they can be reset alongside maps.
        if (!d_write_heads_ || max_write_heads_experts_ < num_experts)
        {
            if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_write_heads_),
                                     MoEWorkspaceBuffers::GROUP_WRITE_HEADS,
                                     static_cast<size_t>(num_experts) * sizeof(int),
                                     "groupTokensByExpertDevice(write_heads)"))
            {
                d_write_heads_ = nullptr;
                max_write_heads_experts_ = 0;
                return false;
            }
            max_write_heads_experts_ = num_experts;
        }
        if (!hipMoE_initialize_grouping({
                .expert_counts = d_expert_counts,
                .write_heads = d_write_heads_,
                .original_to_grouped = d_group_original_to_grouped_,
                .grouped_token_indices = d_grouped_token_indices,
                .grouped_weights = d_grouped_weights,
                .num_experts = num_experts,
                .total_slots = total_slots}, device_ordinal_, stream))
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] grouping initialization failed");
            return false;
        }

        // Step 2: Count per expert
        if (!hipMoE_count_per_expert(d_routing_indices, d_expert_counts,
                                     total_slots, num_experts,
                                     device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] count_per_expert failed");
            return false;
        }

        // Step 3: Exclusive scan (expert_counts → expert_offsets)
        if (!hipMoE_exclusive_scan(d_expert_counts, d_expert_offsets,
                                   num_experts,
                                   device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] exclusive_scan failed");
            return false;
        }

        if (packed ? !hipMoE_build_active_expert_list_with_offset_slots(*packed) :
            (d_group_active_expert_ids_ && !hipMoE_build_active_expert_list(
                d_expert_counts, d_group_active_expert_ids_,
                num_experts, std::min(total_slots, num_experts),
                device_ordinal_, getStream())))
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] active expert list failed");
            return false;
        }

        // Scatter consumes the cursors reset before counting on this stream.
        if (!hipMoE_scatter_tokens(d_routing_indices, d_routing_weights,
                                   d_expert_offsets, d_write_heads_,
                                   d_grouped_token_indices,
                                   d_group_original_to_grouped_,
                                   d_grouped_weights,
                                   total_slots, num_experts, top_k,
                                   device_ordinal_, getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::groupTokensByExpertDevice] scatter_tokens failed");
            return false;
        }

        return true;
    }

    // =========================================================================
    // Tensor-aware API overrides — GPU implementations
    //
    // These use gpu_data_ptr() for device-resident data and keep
    // all computation on device.  Small host-side arrays (token indices,
    // routing weights) are uploaded via cached staging buffers.
    // =========================================================================

    bool ROCmMoEKernel::ensureStagingCapacity(int count)
    {
        if (count <= staging_capacity_)
            return d_staging_indices_ && d_staging_weights_;

        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_staging_indices_),
                                 MoEWorkspaceBuffers::STAGING_INDICES,
                                 static_cast<size_t>(count) * sizeof(int),
                                 "ensureStagingCapacity(indices)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_staging_weights_),
                                 MoEWorkspaceBuffers::STAGING_WEIGHTS,
                                 static_cast<size_t>(count) * sizeof(float),
                                 "ensureStagingCapacity(weights)"))
        {
            d_staging_indices_ = nullptr;
            d_staging_weights_ = nullptr;
            staging_capacity_ = 0;
            return false;
        }
        staging_capacity_ = count;
        return true;
    }

    bool ROCmMoEKernel::ensureGroupedDecodeCapacity(
        int num_active,
        int intermediate,
        int d_model)
    {
        if (num_active <= 0 || intermediate <= 0 || d_model <= 0 ||
            (intermediate % 32) != 0)
            return false;

        if (!setMoEDevice(device_ordinal_, "ensureGroupedDecodeCapacity"))
            return false;

        const int blocks_per_row = intermediate / 32;
        const bool pointer_capacity_ok = grouped_decode_active_cap_ >= num_active;
        const bool activation_capacity_ok = grouped_decode_intermediate_cap_ >= intermediate;
        const bool output_capacity_ok = grouped_decode_d_model_cap_ >= d_model;
        if (pointer_capacity_ok && activation_capacity_ok && output_capacity_ok &&
            d_grouped_gate_ptrs_ && d_grouped_up_ptrs_ && d_grouped_decode_weights_ &&
            d_grouped_expert_ids_ && d_grouped_down_descs_ &&
            d_grouped_swiglu_int8_ && d_grouped_swiglu_scales_ &&
            d_grouped_down_partials_)
        {
            return true;
        }

        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_gate_ptrs_),
                                 MoEWorkspaceBuffers::ROCM_DECODE_GATE_PTRS,
                                 static_cast<size_t>(num_active) * sizeof(float *),
                                 "ensureGroupedDecodeCapacity(gate_ptrs)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_up_ptrs_),
                                 MoEWorkspaceBuffers::ROCM_DECODE_UP_PTRS,
                                 static_cast<size_t>(num_active) * sizeof(float *),
                                 "ensureGroupedDecodeCapacity(up_ptrs)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_expert_ids_),
                                 MoEWorkspaceBuffers::DECODE_EXPERT_IDS,
                                 static_cast<size_t>(num_active) * sizeof(int),
                                 "ensureGroupedDecodeCapacity(expert_ids)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_decode_weights_),
                                 MoEWorkspaceBuffers::DECODE_WEIGHTS,
                                 static_cast<size_t>(num_active) * sizeof(float),
                                 "ensureGroupedDecodeCapacity(weights)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_down_descs_),
                                 MoEWorkspaceBuffers::ROCM_DECODE_DOWN_DESCS,
                                 static_cast<size_t>(num_active) * sizeof(DeviceNativeQuantisedMatrixDesc),
                                 "ensureGroupedDecodeCapacity(down_descs)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_swiglu_int8_),
                                 MoEWorkspaceBuffers::DECODE_SWIGLU_INT8,
                                 static_cast<size_t>(num_active) * intermediate * sizeof(int8_t),
                                 "ensureGroupedDecodeCapacity(swiglu_int8)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_swiglu_scales_),
                                 MoEWorkspaceBuffers::DECODE_SWIGLU_SCALES,
                                 static_cast<size_t>(num_active) * blocks_per_row * sizeof(float),
                                 "ensureGroupedDecodeCapacity(swiglu_scales)") ||
            !bindWorkspaceBuffer(
                reinterpret_cast<void **>(
                    &d_grouped_down_partials_),
                MoEWorkspaceBuffers::DOWN_PARTIALS,
                static_cast<size_t>(num_active) *
                    static_cast<size_t>(
                        NativeQuantisedGroupedDecodePolicy::maximum_k_partitions) *
                    static_cast<size_t>(d_model) * sizeof(float),
                "ensureGroupedDecodeCapacity(ordered_down_partials)"))
        {
            d_grouped_gate_ptrs_ = nullptr;
            d_grouped_up_ptrs_ = nullptr;
            d_grouped_expert_ids_ = nullptr;
            d_grouped_decode_weights_ = nullptr;
            d_grouped_down_descs_ = nullptr;
            d_grouped_swiglu_int8_ = nullptr;
            d_grouped_swiglu_scales_ = nullptr;
            d_grouped_down_partials_ = nullptr;
            grouped_decode_active_cap_ = 0;
            grouped_decode_intermediate_cap_ = 0;
            grouped_decode_d_model_cap_ = 0;
            return false;
        }

        grouped_decode_active_cap_ = num_active;
        grouped_decode_intermediate_cap_ = intermediate;
        grouped_decode_d_model_cap_ = d_model;
        return true;
    }

    bool ROCmMoEKernel::ensureGroupedGateUpCapacity(
        int num_active,
        int d_model,
        int intermediate)
    {
        if (num_active <= 0 || d_model <= 0 || intermediate <= 0 ||
            (d_model % 32) != 0)
            return false;

        if (!setMoEDevice(device_ordinal_, "ensureGroupedGateUpCapacity"))
            return false;

        const int blocks_per_row = d_model / 32;
        if (grouped_gateup_active_cap_ >= num_active &&
            grouped_gateup_d_model_cap_ >= d_model &&
            grouped_gateup_intermediate_cap_ >= intermediate &&
            d_grouped_gate_output_ptrs_ && d_grouped_up_output_ptrs_ &&
            d_grouped_gateup_expert_ids_ &&
            d_grouped_hidden_int8_ && d_grouped_hidden_scales_ &&
            d_grouped_gateup_gate_partials_ &&
            d_grouped_gateup_up_partials_)
        {
            return true;
        }

        if (!bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_gate_output_ptrs_),
                                 MoEWorkspaceBuffers::ROCM_DECODE_GATE_OUTPUT_PTRS,
                                 static_cast<size_t>(num_active) * sizeof(float *),
                                 "ensureGroupedGateUpCapacity(gate_output_ptrs)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_up_output_ptrs_),
                                 MoEWorkspaceBuffers::ROCM_DECODE_UP_OUTPUT_PTRS,
                                 static_cast<size_t>(num_active) * sizeof(float *),
                                 "ensureGroupedGateUpCapacity(up_output_ptrs)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_gateup_expert_ids_),
                                 MoEWorkspaceBuffers::DECODE_EXPERT_IDS,
                                 static_cast<size_t>(num_active) * sizeof(int),
                                 "ensureGroupedGateUpCapacity(expert_ids)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_hidden_int8_),
                                 MoEWorkspaceBuffers::DECODE_HIDDEN_INT8,
                                 static_cast<size_t>(d_model) * sizeof(int8_t),
                                 "ensureGroupedGateUpCapacity(hidden_int8)") ||
            !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_hidden_scales_),
                                 MoEWorkspaceBuffers::DECODE_HIDDEN_SCALES,
                                 static_cast<size_t>(blocks_per_row) * sizeof(float),
                                 "ensureGroupedGateUpCapacity(hidden_scales)") ||
            !bindWorkspaceBuffer(
                reinterpret_cast<void **>(&d_grouped_gateup_gate_partials_),
                MoEWorkspaceBuffers::GATEUP_GATE_PARTIALS,
                static_cast<size_t>(num_active) *
                    static_cast<size_t>(
                        NativeQuantisedGroupedDecodePolicy::maximum_k_partitions) *
                    static_cast<size_t>(intermediate) * sizeof(float),
                "ensureGroupedGateUpCapacity(gate_partials)") ||
            !bindWorkspaceBuffer(
                reinterpret_cast<void **>(&d_grouped_gateup_up_partials_),
                MoEWorkspaceBuffers::GATEUP_UP_PARTIALS,
                static_cast<size_t>(num_active) *
                    static_cast<size_t>(
                        NativeQuantisedGroupedDecodePolicy::maximum_k_partitions) *
                    static_cast<size_t>(intermediate) * sizeof(float),
                "ensureGroupedGateUpCapacity(up_partials)"))
        {
            d_grouped_gate_output_ptrs_ = nullptr;
            d_grouped_up_output_ptrs_ = nullptr;
            d_grouped_gateup_expert_ids_ = nullptr;
            d_grouped_hidden_int8_ = nullptr;
            d_grouped_hidden_scales_ = nullptr;
            d_grouped_gateup_gate_partials_ = nullptr;
            d_grouped_gateup_up_partials_ = nullptr;
            grouped_gateup_active_cap_ = 0;
            grouped_gateup_d_model_cap_ = 0;
            grouped_gateup_intermediate_cap_ = 0;
            return false;
        }

        grouped_gateup_active_cap_ = num_active;
        grouped_gateup_d_model_cap_ = d_model;
        grouped_gateup_intermediate_cap_ = intermediate;
        return true;
    }

    bool ROCmMoEKernel::isDecodeGraphCaptureActive() const
    {
        const bool capture_active = isGraphCaptureActive() ||
                                    (deviceContext() && deviceContext()->isDeviceGraphCaptureActive());
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
        void *stream = getStream();
        if (stream)
        {
            const hipError_t capture_err = hipStreamIsCapturing(static_cast<hipStream_t>(stream), &capture_status);
            if (capture_err != hipSuccess)
                capture_status = hipStreamCaptureStatusNone;
        }
        return capture_active || capture_status == hipStreamCaptureStatusActive;
    }

    bool ROCmMoEKernel::rejectDecodeStagingDuringCapture(const char *context) const
    {
        if (!isDecodeGraphCaptureActive())
            return false;

        LOG_ERROR("[ROCmMoEKernel] " << context
                                    << " requires pointer/metadata staging during graph capture. "
                                       "Run a warmup execution with the same scratch tensors and workspace first.");
        return true;
    }

    bool ROCmMoEKernel::resolveFixedTableGateUpMetadata(
        std::size_t persistent_descriptor_slot,
        RuntimePointerArrayScope scope,
        const int *expert_ids,
        int num_active,
        const int **device_expert_ids)
    {
        if (!expert_ids || !device_expert_ids || num_active <= 0 ||
            num_active > static_cast<int>(kRuntimePointerArrayMaxTopK))
            return false;

        const bool capture_active = isDecodeGraphCaptureActive();
        std::size_t workspace_slot = 0;
        if (!runtimePointerWorkspaceSlot(
                persistent_descriptor_slot,
                scope,
                MoERuntimePointerArrayRole::GateUp,
                capture_active
                    ? MoERuntimePointerWorkspaceAccess::CaptureExistingOnly
                    : MoERuntimePointerWorkspaceAccess::WarmupMayAcquire,
                &workspace_slot,
                "fixed-table gate/up metadata"))
        {
            return false;
        }

        auto *slot_ids = static_cast<int *>(workspace_->getPersistentSlotBuffer(
            MoEWorkspaceBuffers::FIXED_DECODE_GATEUP_EXPERT_IDS,
            kRuntimePointerArrayWorkspaceEntries,
            workspace_slot,
            static_cast<std::size_t>(num_active) * sizeof(int)));
        if (!slot_ids)
            return false;

        const auto state = fixed_gateup_metadata_states_.find(workspace_slot);
        if (state != fixed_gateup_metadata_states_.end())
        {
            const bool matches = state->second.num_active == num_active &&
                                 std::equal(
                                     state->second.expert_ids.begin(),
                                     state->second.expert_ids.begin() + num_active,
                                     expert_ids);
            if (!matches)
            {
                LOG_ERROR("[ROCmMoEKernel] Fixed-table gate/up metadata is immutable "
                          "for one captured owner slot; use the device-routed API "
                          "for changing expert ids");
                return false;
            }
            *device_expert_ids = slot_ids;
            return true;
        }
        if (capture_active)
        {
            LOG_ERROR("[ROCmMoEKernel] Fixed-table gate/up metadata was not "
                      "published before graph capture");
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "resolveFixedTableGateUpMetadata"))
            return false;

        const hipError_t err = hipMemcpyAsync(
            slot_ids,
            expert_ids,
            static_cast<std::size_t>(num_active) * sizeof(int),
            hipMemcpyHostToDevice,
            static_cast<hipStream_t>(getStream()));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel] Fixed-table gate/up metadata publication failed: "
                      << hipGetErrorString(err));
            return false;
        }

        FixedGateUpMetadataState published;
        published.num_active = num_active;
        std::copy(expert_ids, expert_ids + num_active, published.expert_ids.begin());
        fixed_gateup_metadata_states_.emplace(workspace_slot, std::move(published));
        *device_expert_ids = slot_ids;
        return true;
    }

    bool ROCmMoEKernel::resolveFixedTableDownMetadata(
        std::size_t persistent_descriptor_slot,
        RuntimePointerArrayScope scope,
        const int *expert_ids,
        const float *expert_weights,
        int num_active,
        const int **device_expert_ids,
        const float **device_expert_weights)
    {
        if (!expert_ids || !expert_weights || !device_expert_ids ||
            !device_expert_weights || num_active <= 0 ||
            num_active > static_cast<int>(kRuntimePointerArrayMaxTopK))
            return false;

        const bool capture_active = isDecodeGraphCaptureActive();
        std::size_t workspace_slot = 0;
        if (!runtimePointerWorkspaceSlot(
                persistent_descriptor_slot,
                scope,
                MoERuntimePointerArrayRole::Down,
                capture_active
                    ? MoERuntimePointerWorkspaceAccess::CaptureExistingOnly
                    : MoERuntimePointerWorkspaceAccess::WarmupMayAcquire,
                &workspace_slot,
                "fixed-table down metadata"))
        {
            return false;
        }

        auto *slot_ids = static_cast<int *>(workspace_->getPersistentSlotBuffer(
            MoEWorkspaceBuffers::FIXED_DECODE_DOWN_EXPERT_IDS,
            kRuntimePointerArrayWorkspaceEntries,
            workspace_slot,
            static_cast<std::size_t>(num_active) * sizeof(int)));
        auto *slot_weights = static_cast<float *>(workspace_->getPersistentSlotBuffer(
            MoEWorkspaceBuffers::FIXED_DECODE_DOWN_WEIGHTS,
            kRuntimePointerArrayWorkspaceEntries,
            workspace_slot,
            static_cast<std::size_t>(num_active) * sizeof(float)));
        if (!slot_ids || !slot_weights)
            return false;

        const auto state = fixed_down_metadata_states_.find(workspace_slot);
        if (state != fixed_down_metadata_states_.end())
        {
            const bool matches =
                state->second.num_active == num_active &&
                std::equal(
                    state->second.expert_ids.begin(),
                    state->second.expert_ids.begin() + num_active,
                    expert_ids) &&
                std::equal(
                    state->second.expert_weights.begin(),
                    state->second.expert_weights.begin() + num_active,
                    expert_weights);
            if (!matches)
            {
                LOG_ERROR("[ROCmMoEKernel] Fixed-table down metadata is immutable "
                          "for one captured owner slot; use the device-routed API "
                          "for changing expert ids or weights");
                return false;
            }
            *device_expert_ids = slot_ids;
            *device_expert_weights = slot_weights;
            return true;
        }
        if (capture_active)
        {
            LOG_ERROR("[ROCmMoEKernel] Fixed-table down metadata was not "
                      "published before graph capture");
            return false;
        }
        if (!setMoEDevice(device_ordinal_, "resolveFixedTableDownMetadata"))
            return false;

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipMemcpyAsync(
            slot_ids,
            expert_ids,
            static_cast<std::size_t>(num_active) * sizeof(int),
            hipMemcpyHostToDevice,
            stream);
        if (err == hipSuccess)
        {
            err = hipMemcpyAsync(
                slot_weights,
                expert_weights,
                static_cast<std::size_t>(num_active) * sizeof(float),
                hipMemcpyHostToDevice,
                stream);
        }
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmMoEKernel] Fixed-table down metadata publication failed: "
                      << hipGetErrorString(err));
            return false;
        }

        FixedDownMetadataState published;
        published.num_active = num_active;
        std::copy(expert_ids, expert_ids + num_active, published.expert_ids.begin());
        std::copy(
            expert_weights,
            expert_weights + num_active,
            published.expert_weights.begin());
        fixed_down_metadata_states_.emplace(workspace_slot, std::move(published));
        *device_expert_ids = slot_ids;
        *device_expert_weights = slot_weights;
        return true;
    }

    bool ROCmMoEKernel::runtimePointerWorkspaceSlot(
        std::size_t persistent_descriptor_slot,
        RuntimePointerArrayScope scope,
        MoERuntimePointerArrayRole role,
        MoERuntimePointerWorkspaceAccess access,
        std::size_t *workspace_slot,
        const char *context)
    {
        if (!workspace_slot)
            return false;

        const auto slot = runtime_pointer_workspace_owners_.resolve(
            workspace_,
            role,
            persistent_descriptor_slot,
            static_cast<std::size_t>(scope),
            access,
            context);
        if (!slot || *slot >= kRuntimePointerArrayWorkspaceEntries)
        {
            LOG_ERROR("[ROCmMoEKernel] "
                      << (context ? context : "runtime pointer table")
                      << " could not resolve an exclusive pointer workspace slot");
            return false;
        }

        *workspace_slot = *slot;
        return true;
    }

    /**
     * @brief Bind and publish the table-owned gate/up pointer arrays.
     * @param persistent_descriptor_slot Descriptor lifetime owning this slot.
     * @param scope Distinct execution use of the descriptor table.
     * @param top_k Number of active pointers to publish.
     * @param gate_ptrs Prepared gate output addresses on this device.
     * @param up_ptrs Prepared up output addresses on this device.
     * @param d_gate_ptrs Receives the immutable device gate pointer-array address.
     * @param d_up_ptrs Receives the immutable device up pointer-array address.
     * @return True when warmup publication or capture-time reuse is complete.
     *
     * Pointer publication owns its arena binding, independently of quantized
     * projection scratch. Capture may only consume an already prepared slot.
     */
    bool ROCmMoEKernel::stageRuntimeGateUpPointerArrays(
        std::size_t persistent_descriptor_slot,
        RuntimePointerArrayScope scope,
        int top_k,
        const std::array<float *, ROCmMoEKernel::kRuntimePointerArrayMaxTopK> &gate_ptrs,
        const std::array<float *, ROCmMoEKernel::kRuntimePointerArrayMaxTopK> &up_ptrs,
        float ***d_gate_ptrs,
        float ***d_up_ptrs)
    {
        if (!d_gate_ptrs || !d_up_ptrs || top_k <= 0 ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK))
        {
            return false;
        }

        const bool capture_active = isDecodeGraphCaptureActive();
        std::size_t workspace_slot = 0;
        if (!runtimePointerWorkspaceSlot(
                persistent_descriptor_slot,
                scope,
                MoERuntimePointerArrayRole::GateUp,
                capture_active
                    ? MoERuntimePointerWorkspaceAccess::CaptureExistingOnly
                    : MoERuntimePointerWorkspaceAccess::WarmupMayAcquire,
                &workspace_slot,
                "grouped gate/up"))
            return false;
        if (!setMoEDevice(device_ordinal_, "stageRuntimeGateUpPointerArrays"))
            return false;

        if (!d_grouped_gate_output_ptrs_ || !d_grouped_up_output_ptrs_)
        {
            const size_t bytes = kRuntimePointerArrayWorkspaceEntries *
                                 kRuntimePointerArrayMaxTopK * sizeof(float *);
            if (capture_active ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_gate_output_ptrs_),
                    MoEWorkspaceBuffers::ROCM_DECODE_GATE_OUTPUT_PTRS, bytes,
                    "stageRuntimeGateUpPointerArrays(gate)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_up_output_ptrs_),
                    MoEWorkspaceBuffers::ROCM_DECODE_UP_OUTPUT_PTRS, bytes,
                    "stageRuntimeGateUpPointerArrays(up)"))
            {
                LOG_ERROR("[ROCmMoEKernel::stageRuntimeGateUpPointerArrays] Grouped gate/up workspace is not prepared");
                return false;
            }
        }

        float **slot_gate_ptrs =
            d_grouped_gate_output_ptrs_ + workspace_slot * kRuntimePointerArrayMaxTopK;
        float **slot_up_ptrs =
            d_grouped_up_output_ptrs_ + workspace_slot * kRuntimePointerArrayMaxTopK;

        if (capture_active)
        {
            if (!gateup_pointer_slot_ready_[workspace_slot])
                return rejectDecodeStagingDuringCapture("grouped gate/up pointer workspace slot");
            *d_gate_ptrs = slot_gate_ptrs;
            *d_up_ptrs = slot_up_ptrs;
            return true;
        }

        for (int slot = 0; slot < top_k; ++slot)
        {
            if (!gate_ptrs[slot] || !up_ptrs[slot])
            {
                LOG_ERROR("[ROCmMoEKernel::stageRuntimeGateUpPointerArrays] Null gate/up output pointer for slot "
                          << slot << " persistent_descriptor_slot="
                          << persistent_descriptor_slot);
                return false;
            }
        }

        if (!hipMoE_stage_mutable_pointer_arrays(
                slot_gate_ptrs,
                slot_up_ptrs,
                gate_ptrs.data(),
                up_ptrs.data(),
                top_k,
                device_ordinal_,
                getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::stageRuntimeGateUpPointerArrays] Pointer staging failed");
            return false;
        }

        gateup_pointer_slot_ready_[workspace_slot] = true;
        *d_gate_ptrs = slot_gate_ptrs;
        *d_up_ptrs = slot_up_ptrs;
        return true;
    }

    /**
     * @brief Bind and publish table-owned inputs without requiring Q8 scratch.
     * @param persistent_descriptor_slot Descriptor lifetime owning this slot.
     * @param scope Distinct execution use of the descriptor table.
     * @param top_k Number of active pointers to publish.
     * @param gate_ptrs Prepared gate inputs on this device.
     * @param up_ptrs Prepared up inputs on this device.
     * @param d_gate_ptrs Receives the immutable device gate pointer-array address.
     * @param d_up_ptrs Receives the immutable device up pointer-array address.
     * @return True when warmup publication or capture-time reuse is complete.
     */
    bool ROCmMoEKernel::stageRuntimeDownPointerArrays(
        std::size_t persistent_descriptor_slot,
        RuntimePointerArrayScope scope,
        int top_k,
        const std::array<const float *, ROCmMoEKernel::kRuntimePointerArrayMaxTopK> &gate_ptrs,
        const std::array<const float *, ROCmMoEKernel::kRuntimePointerArrayMaxTopK> &up_ptrs,
        const float ***d_gate_ptrs,
        const float ***d_up_ptrs)
    {
        if (!d_gate_ptrs || !d_up_ptrs || top_k <= 0 ||
            top_k > static_cast<int>(kRuntimePointerArrayMaxTopK))
        {
            return false;
        }

        const bool capture_active = isDecodeGraphCaptureActive();
        std::size_t workspace_slot = 0;
        if (!runtimePointerWorkspaceSlot(
                persistent_descriptor_slot,
                scope,
                MoERuntimePointerArrayRole::Down,
                capture_active
                    ? MoERuntimePointerWorkspaceAccess::CaptureExistingOnly
                    : MoERuntimePointerWorkspaceAccess::WarmupMayAcquire,
                &workspace_slot,
                "grouped down"))
            return false;
        if (!setMoEDevice(device_ordinal_, "stageRuntimeDownPointerArrays"))
            return false;

        if (!d_grouped_gate_ptrs_ || !d_grouped_up_ptrs_)
        {
            const size_t bytes = kRuntimePointerArrayWorkspaceEntries *
                                 kRuntimePointerArrayMaxTopK * sizeof(float *);
            if (capture_active ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_gate_ptrs_),
                    MoEWorkspaceBuffers::ROCM_DECODE_GATE_PTRS, bytes,
                    "stageRuntimeDownPointerArrays(gate)") ||
                !bindWorkspaceBuffer(reinterpret_cast<void **>(&d_grouped_up_ptrs_),
                    MoEWorkspaceBuffers::ROCM_DECODE_UP_PTRS, bytes,
                    "stageRuntimeDownPointerArrays(up)"))
            {
                LOG_ERROR("[ROCmMoEKernel::stageRuntimeDownPointerArrays] Grouped down workspace is not prepared");
                return false;
            }
        }

        const float **slot_gate_ptrs =
            d_grouped_gate_ptrs_ + workspace_slot * kRuntimePointerArrayMaxTopK;
        const float **slot_up_ptrs =
            d_grouped_up_ptrs_ + workspace_slot * kRuntimePointerArrayMaxTopK;

        if (capture_active)
        {
            if (!down_pointer_slot_ready_[workspace_slot])
                return rejectDecodeStagingDuringCapture("grouped down pointer workspace slot");
            *d_gate_ptrs = slot_gate_ptrs;
            *d_up_ptrs = slot_up_ptrs;
            return true;
        }

        for (int slot = 0; slot < top_k; ++slot)
        {
            if (!gate_ptrs[slot] || !up_ptrs[slot])
            {
                LOG_ERROR("[ROCmMoEKernel::stageRuntimeDownPointerArrays] Null gate/up pointer for slot "
                          << slot << " persistent_descriptor_slot="
                          << persistent_descriptor_slot);
                return false;
            }
        }

        if (!hipMoE_stage_const_pointer_arrays(
                slot_gate_ptrs,
                slot_up_ptrs,
                gate_ptrs.data(),
                up_ptrs.data(),
                top_k,
                device_ordinal_,
                getStream()))
        {
            LOG_ERROR("[ROCmMoEKernel::stageRuntimeDownPointerArrays] Pointer staging failed");
            return false;
        }

        down_pointer_slot_ready_[workspace_slot] = true;
        *d_gate_ptrs = slot_gate_ptrs;
        *d_up_ptrs = slot_up_ptrs;
        return true;
    }

    bool ROCmMoEKernel::routeWithTensorsImpl(
        ITensor *hidden, ITensor *gate_weights,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        MoERoutingResult &host_result,
        const int *device_effective_seq_len,
        const char *context)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        const float *h = static_cast<const float *>(hidden->gpu_data_ptr());
        const void *g = gate_weights->gpu_data_ptr();
        const TensorType gate_type = gate_weights->native_type();
        if (!h || !g)
        {
            LOG_ERROR("[" << context << "] null device pointer "
                      "(hidden="
                      << (const void *)h << " gate=" << (const void *)g << ")");
            return false;
        }

        hipStream_t stream = static_cast<hipStream_t>(getStream());
        if (!stream)
        {
            LOG_ERROR("[" << context << "] explicit HIP stream is required");
            return false;
        }
        float *d_idx = static_cast<float *>(output_indices->gpu_data_ptr());
        float *d_wt = static_cast<float *>(output_weights->gpu_data_ptr());
        if (!d_idx || !d_wt)
        {
            LOG_ERROR("[" << context << "] output tensors have no device allocation");
            return false;
        }

        if (!routeCore(h, g, gate_type, seq_len, d_model, num_experts, top_k,
                       normalize_weights, d_idx, d_wt,
                       device_effective_seq_len))
            return false;

        host_result.expert_indices.clear();
        host_result.expert_weights.clear();
        host_result.router_logits.clear();

        /*
         * Routing tensors remain device-authoritative in eager execution and
         * graph replay alike.  Snapshot infrastructure performs any requested
         * D2H materialization after publication, at an explicit observation
         * boundary, rather than maintaining a live host mirror here.
         */
        const auto device = DeviceId::rocm(device_ordinal_);
        markDeviceWritten(output_indices, device, getStream());
        markDeviceWritten(output_weights, device, getStream());

        return true;
    }

    bool ROCmMoEKernel::routeWithTensors(
        ITensor *hidden, ITensor *gate_weights,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        MoERoutingResult &host_result)
    {
        return routeWithTensorsImpl(hidden, gate_weights,
                                    seq_len, d_model, num_experts, top_k,
                                    normalize_weights,
                                    output_indices, output_weights,
                                    host_result,
                                    nullptr,
                                    "ROCmMoEKernel::routeWithTensors");
    }

    bool ROCmMoEKernel::routeWithTensorsEffectiveSeqLen(
        ITensor *hidden, ITensor *gate_weights,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        MoERoutingResult &host_result,
        const int *device_effective_seq_len)
    {
        if (!device_effective_seq_len)
        {
            LOG_ERROR("[ROCmMoEKernel::routeWithTensorsEffectiveSeqLen] missing device effective length scalar");
            return false;
        }
        return routeWithTensorsImpl(hidden, gate_weights,
                                    seq_len, d_model, num_experts, top_k,
                                    normalize_weights,
                                    output_indices, output_weights,
                                    host_result,
                                    device_effective_seq_len,
                                    "ROCmMoEKernel::routeWithTensorsEffectiveSeqLen");
    }

    bool ROCmMoEKernel::publishOwnedRouteRows(const MoERouterRowPublication &publication)
    {
        return rocm::publishRouterOwnedRows(publication, getStream());
    }

    bool ROCmMoEKernel::routeOwnedRowsWithTensors(
        ITensor *hidden, ITensor *gate_weights,
        const MoERouterRowPacketLayout &layout, int d_model, int num_experts,
        bool normalize_weights, ITensor *packet, std::uint64_t *selected_bytes,
        const std::int32_t *live_rows)
    {
        constexpr const char *context = "ROCmMoEKernel::routeOwnedRowsWithTensors";
        void *stream = getStream();
        const auto device = DeviceId::rocm(device_ordinal_);
        if (!stream || !layout.valid() || d_model <= 0 || num_experts < layout.top_k ||
            !live_rows || !selected_bytes || !hidden || !gate_weights || !packet ||
            hidden->native_type() != TensorType::FP32 ||
            hidden->numel() < std::size_t(layout.capacity) * d_model ||
            gate_weights->numel() < std::size_t(num_experts) * d_model ||
            packet->size_bytes() < layout.packetBytes() ||
            !requireTensorOnDevice(hidden, device, stream, "hidden", context) ||
            !requireTensorOnDevice(gate_weights, device, stream, "gate_weights", context) ||
            !requireOutputOnDevice(packet, device, stream, "router packet", context) ||
            !validateDevicePointerOrLog(selected_bytes, device_ordinal_, "packet extent", context))
            return false;
        const MoERouterOwnedRowOutputs owned{layout.partition,
            static_cast<MoERouterSelectedRoute *>(packet->gpu_data_ptr()), selected_bytes};
        if (!routeCore(static_cast<const float *>(hidden->gpu_data_ptr()), gate_weights->gpu_data_ptr(),
                       gate_weights->native_type(), layout.capacity, d_model, num_experts, layout.top_k,
                       normalize_weights, nullptr, nullptr, live_rows, &owned))
            return false;
        markDeviceWritten(packet, device, stream);
        return true;
    }

    bool ROCmMoEKernel::prepareRouteLaunch(
        ITensor *gate_weights,
        const MoERouteLaunchPlan &plan)
    {
        constexpr const char *kContext = "ROCmMoEKernel::prepareRouteLaunch";
        if (!setMoEDevice(device_ordinal_, kContext))
            return false;
        hipStream_t stream = static_cast<hipStream_t>(getStream());
        if (!stream)
        {
            LOG_ERROR("[" << kContext << "] explicit HIP stream is required");
            return false;
        }
        if (!plan.valid() ||
            (plan.kind == MoERouteLaunchKind::RuntimeDecode &&
             plan.physical_rows != 1))
        {
            LOG_ERROR("[" << kContext << "] invalid route launch plan"
                      << " kind=" << static_cast<int>(plan.kind)
                      << " rows=" << plan.physical_rows
                      << " d_model=" << plan.d_model
                      << " num_experts=" << plan.num_experts
                      << " top_k=" << plan.top_k);
            return false;
        }
        if (plan.kind == MoERouteLaunchKind::RuntimeDecode)
            prepared_decode_resident_set_policy_ = plan.resident_set_policy;
        const DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!requireTensorOnDevice(
                gate_weights,
                device,
                stream,
                "gate_weights",
                kContext))
        {
            return false;
        }
        if (gate_weights->native_type() != TensorType::FP32 &&
            gate_weights->native_type() != TensorType::FP16 &&
            gate_weights->native_type() != TensorType::BF16)
        {
            LOG_ERROR("[" << kContext << "] router gate must be FP32, FP16 or BF16, got "
                      << tensorTypeName(gate_weights->native_type()));
            return false;
        }

        const size_t required_gate =
            static_cast<size_t>(plan.num_experts) *
            static_cast<size_t>(plan.d_model);
        if (gate_weights->numel() < required_gate ||
            !gate_weights->gpu_data_ptr())
        {
            LOG_ERROR("[" << kContext << "] router gate capacity or device pointer is invalid"
                      << " elements=" << gate_weights->numel()
                      << "/" << required_gate
                      << " device_ptr=" << gate_weights->gpu_data_ptr());
            return false;
        }

        if (!prepareRequiredRouterQ8Hidden(plan.physical_rows, plan.d_model, plan.top_k))
            return false;

        const size_t logits_count =
            static_cast<size_t>(plan.physical_rows) *
            static_cast<size_t>(plan.num_experts);
        if (!ensureRouteBufferCapacity(logits_count))
        {
            LOG_ERROR("[" << kContext << "] failed to bind persistent route scratch");
            return false;
        }

        if (gate_weights->native_type() != TensorType::FP32)
            return true;

        const auto &rocm_env = debugEnv().rocm;
        if (plan.kind == MoERouteLaunchKind::DecodeEquivalentVerifier &&
            rocm_env.moe_router_kpart_decode)
        {
            LOG_ERROR("[" << kContext << "] decode-equivalent verifier routing "
                      "cannot be prepared while k-part decode routing is selected");
            return false;
        }

        const auto *gate_device =
            static_cast<const float *>(gate_weights->gpu_data_ptr());
        if (rocm_env.moe_router_q8)
        {
            if ((plan.d_model % 32) != 0)
            {
                LOG_ERROR("[" << kContext << "] Q8 routing requires d_model "
                          "to be a multiple of 32, got " << plan.d_model);
                return false;
            }
            if (!ensureRouterQ8HiddenScratchCapacity(
                    plan.physical_rows,
                    plan.d_model))
            {
                LOG_ERROR("[" << kContext << "] failed to bind persistent Q8 hidden-row scratch");
                return false;
            }
            if (!getOrCreateQ8RouterGateCache(
                    gate_device,
                    plan.d_model,
                    plan.num_experts))
            {
                LOG_ERROR("[" << kContext << "] failed to publish immutable Q8 router gate");
                return false;
            }
            return true;
        }

        if (plan.kind == MoERouteLaunchKind::RuntimeDecode &&
            rocm_env.moe_router_kpart_decode)
        {
            if (rocm_env.moe_router_kparts <= 0 ||
                !ensureRouteLogitsPartialsCapacity(
                    static_cast<size_t>(plan.num_experts) *
                    static_cast<size_t>(rocm_env.moe_router_kparts)))
            {
                LOG_ERROR("[" << kContext << "] failed to bind persistent k-part route scratch");
                return false;
            }
            return true;
        }

        return true;
    }

    bool ROCmMoEKernel::routeVerifierRowsDecodeEquivalent(
        ITensor *hidden, ITensor *gate_weights,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        const int *device_effective_seq_len,
        DeviceMoELayerRuntime *deferred_selected_route_ledger)
    {
        return routeGroupedVerifierWithPolicy(LocalGPUExpertKernelPolicy::Generic,hidden,gate_weights,
            seq_len,d_model,num_experts,top_k,normalize_weights,output_indices,output_weights,
            device_effective_seq_len,deferred_selected_route_ledger);
    }

    bool ROCmMoEKernel::routeGroupedVerifierWithPolicy(LocalGPUExpertKernelPolicy policy,
        ITensor *hidden, ITensor *gate_weights,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        const int *device_effective_seq_len,
        DeviceMoELayerRuntime *deferred_selected_route_ledger)
    {
        constexpr const char *kContext = "ROCmMoEKernel::routeVerifierRowsDecodeEquivalent";
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (!setMoEDevice(device_ordinal_, kContext))
            return false;
        hipStream_t stream = static_cast<hipStream_t>(getStream());
        if (!stream)
        {
            LOG_ERROR("[" << kContext << "] explicit HIP stream is required");
            return false;
        }

        if (seq_len < 1 || d_model <= 0 ||
            num_experts <= 0 || top_k <= 0 || top_k > num_experts)
        {
            LOG_ERROR("[" << kContext << "] invalid verifier routing shape seq_len="
                          << seq_len << " d_model=" << d_model
                          << " num_experts=" << num_experts
                          << " top_k=" << top_k);
            return false;
        }

        const DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!requireTensorOnDevice(hidden, device, stream, "hidden", kContext) ||
            !requireTensorOnDevice(gate_weights, device, stream, "gate_weights", kContext) ||
            !requireOutputOnDevice(output_indices, device, stream, "output_indices", kContext) ||
            !requireOutputOnDevice(output_weights, device, stream, "output_weights", kContext))
        {
            return false;
        }

        if (hidden->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[" << kContext << "] hidden tensor must be FP32, got "
                          << tensorTypeName(hidden->native_type()));
            return false;
        }
        if (output_indices->native_type() != TensorType::FP32 ||
            output_weights->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[" << kContext << "] routing outputs must be FP32 tensors");
            return false;
        }

        const size_t required_hidden =
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
        const size_t required_gate =
            static_cast<size_t>(num_experts) * static_cast<size_t>(d_model);
        const size_t required_topk =
            static_cast<size_t>(seq_len) * static_cast<size_t>(top_k);
        if (hidden->numel() < required_hidden ||
            gate_weights->numel() < required_gate ||
            output_indices->numel() < required_topk ||
            output_weights->numel() < required_topk)
        {
            LOG_ERROR("[" << kContext << "] tensor capacity is too small"
                          << " hidden=" << hidden->numel() << "/" << required_hidden
                          << " gate=" << gate_weights->numel() << "/" << required_gate
                          << " indices=" << output_indices->numel() << "/" << required_topk
                          << " weights=" << output_weights->numel() << "/" << required_topk);
            return false;
        }

        const float *d_hidden = static_cast<const float *>(hidden->gpu_data_ptr());
        const void *d_gate = gate_weights->gpu_data_ptr();
        float *d_idx = static_cast<float *>(output_indices->gpu_data_ptr());
        float *d_wt = static_cast<float *>(output_weights->gpu_data_ptr());
        if (!d_hidden || !d_gate || !d_idx || !d_wt)
        {
            LOG_ERROR("[" << kContext << "] null device pointer hidden="
                          << static_cast<const void *>(d_hidden)
                          << " gate=" << d_gate
                          << " indices=" << static_cast<void *>(d_idx)
                          << " weights=" << static_cast<void *>(d_wt));
            return false;
        }

        const auto &rocm_env = debugEnv().rocm;
        const bool gate_is_fp32 =
            gate_weights->native_type() == TensorType::FP32;
        const bool qualified_default=usesGFX906MI50Kernels(policy) &&
            d_model==2560 && num_experts==512 && top_k==10 &&
            gate_weights->native_type()==TensorType::FP16;
        const bool q8_router_requested = gate_is_fp32 && rocm_env.moe_router_q8;
        if (q8_router_requested && (d_model % 32) != 0)
        {
            LOG_ERROR("[" << kContext << "] Q8 verifier router requires d_model to be a multiple of 32, got "
                          << d_model);
            return false;
        }
        if (rocm_env.moe_router_kpart_decode)
        {
            LOG_ERROR("[" << kContext << "] ROCm k-part decode router is enabled, but verifier-row "
                          "publication does not yet have a k-part row contract");
            return false;
        }

        const size_t logits_count =
            static_cast<size_t>(seq_len) * static_cast<size_t>(num_experts);
        if (!ensureRouteBufferCapacity(logits_count))
        {
            LOG_ERROR("[" << kContext << "] route scratch allocation failed");
            return false;
        }

        bool used_q8_grouped_router = false;
        bool used_fp16_grouped_router = false;
        invalidateRouterQ8HiddenPublication();
        if (!prepareRequiredRouterQ8Hidden(seq_len, d_model, top_k))
            return false;
        if (!q8_router_requested && !publishRequiredRouterQ8Hidden(
                d_hidden, seq_len, d_model, top_k, device_effective_seq_len))
            return false;
        if (q8_router_requested)
        {
            if (!ensureRouterQ8HiddenScratchCapacity(seq_len, d_model))
            {
                LOG_ERROR("[" << kContext << "] Q8 router hidden scratch unavailable for d_model="
                              << d_model);
                return false;
            }
            const auto *q8_gate = getOrCreateQ8RouterGateCache(
                static_cast<const float *>(d_gate), d_model, num_experts);
            if (!q8_gate)
            {
                LOG_ERROR("[" << kContext << "] Q8 router gate cache unavailable");
                return false;
            }
            if (!hipMoE_gate_logits_q8_weights_decode_equivalent_rows(
                    d_hidden,
                    d_router_q8_hidden_,
                    d_router_q8_hidden_scales_,
                    q8_gate->d_gate_weights_q8,
                    q8_gate->d_gate_scales,
                    d_route_logits_,
                    seq_len,
                    d_model,
                    num_experts,
                    device_ordinal_,
                    getStream(),
                    device_effective_seq_len, routerQ8Basis(debugEnv().gemm.moe_router_arithmetic)))
            {
                LOG_ERROR("[" << kContext << "] grouped Q8 verifier logits failed");
                return false;
            }
            used_q8_grouped_router = true;
            publishRouterQ8Hidden(
                d_hidden,
                seq_len,
                isDecodeGraphCaptureActive(),
                routerQ8Basis(debugEnv().gemm.moe_router_arithmetic));
        }
        else if (gate_is_fp32)
        {
            if (!launchDecodeEquivalentFP32Rows(d_hidden, static_cast<const float *>(d_gate),
                    d_route_logits_, seq_len, d_model, num_experts, device_ordinal_,
                    getStream(), device_effective_seq_len))
            {
                LOG_ERROR("[" << kContext << "] grouped FP32 verifier logits failed");
                return false;
            }
        }
        else if (gate_weights->native_type() == TensorType::FP16)
        {
            if (!hipMoE_gate_logits_fp16_decode_equivalent_rows_with_policy(
                    d_hidden,
                    d_gate,
                    d_route_logits_,
                    seq_len,
                    d_model,
                    num_experts,
                    device_ordinal_,
                    getStream(),
                    device_effective_seq_len,qualified_default))
            {
                LOG_ERROR("[" << kContext << "] grouped FP16 verifier logits failed");
                return false;
            }
            used_fp16_grouped_router = true;
        }
        else if (gate_weights->native_type() == TensorType::BF16)
        {
            if (!hipMoE_gate_logits_bf16_decode_equivalent_rows(
                    d_hidden,
                    d_gate,
                    d_route_logits_,
                    seq_len,
                    d_model,
                    num_experts,
                    device_ordinal_,
                    getStream(),
                    device_effective_seq_len))
            {
                LOG_ERROR("[" << kContext << "] grouped BF16 verifier logits failed");
                return false;
            }
        }
        else
        {
            LOG_ERROR("[" << kContext << "] unsupported router gate dtype "
                          << tensorTypeName(gate_weights->native_type()));
            return false;
        }

        if (!hipMoE_softmax_topk_decode_equivalent_rows_with_policy(
                d_route_logits_,
                d_idx,
                d_wt,
                seq_len,
                num_experts,
                top_k,
                normalize_weights,
                device_ordinal_,
                getStream(),
                device_effective_seq_len,
                static_cast<void *>(deferred_selected_route_ledger),qualified_default))
        {
            LOG_ERROR("[" << kContext << "] grouped decode-equivalent softmax/top-k failed");
            return false;
        }

        markDeviceWritten(output_indices, device, getStream());
        markDeviceWritten(output_weights, device, getStream());
        PerfStatsCollector::addCounter(
            "kernel",
            "rocm_moe_decode_equivalent_runtime_m_router_calls",
            1.0,
            {},
            {},
            {{"seq_len", std::to_string(seq_len)},
             {"d_model", std::to_string(d_model)},
             {"num_experts", std::to_string(num_experts)},
             {"top_k", std::to_string(top_k)},
             {"row_tile", "16"},
             {"tile_count", std::to_string((seq_len + 15) / 16)},
             {"route", used_q8_grouped_router ? "grouped_decode_equivalent_q8"
                                               : (used_fp16_grouped_router
                                                      ? "grouped_decode_equivalent_fp16"
                                                      : "grouped_decode_equivalent")}});
        return true;
    }

    bool ROCmMoEKernel::decodeRouteSelect(
        DeviceMoELayerRuntime *runtime_layer,
        ITensor *hidden, ITensor *gate_weights,
        int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        bool write_legacy_outputs,
        bool update_runtime_histogram,
        const int32_t *absolute_position_ids_device,
        RoutedExpertRowExecutionPolicy row_execution_policy,
        const int32_t *active_row_count_device)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (!runtime_layer || !hidden || !gate_weights)
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] null runtime/input/gate tensor");
            return false;
        }
        if (d_model <= 0 || num_experts <= 0 || top_k <= 0 || top_k > num_experts)
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] invalid dimensions d_model=" << d_model
                                                                                       << " num_experts=" << num_experts
                                                                                       << " top_k=" << top_k);
            return false;
        }

        if (paired_router_descriptors_)
        {
            requirePairedRouterDescriptors(runtime_layer, num_experts, top_k);
            if (row_execution_policy != RoutedExpertRowExecutionPolicy::IntermediateShardLocal)
                throw std::logic_error("Paired descriptor router changed its original local-fragment policy");
        }

        // The runtime table is a graph-build invariant validated by the MoE
        // stages before decode capture.  Do not copy it back here: this path is
        // part of vLLM-style speculative decode replay and must stay entirely
        // device-resident once the graph is hot.

        const float *h = static_cast<const float *>(hidden->gpu_data_ptr());
        const void *g = gate_weights->gpu_data_ptr();
        const TensorType gate_type = gate_weights->native_type();
        if (!h || !g)
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] null device pointer "
                      "(hidden="
                      << (const void *)h << " gate=" << (const void *)g << ")");
            return false;
        }

        if (!ensureRouteBufferCapacity(static_cast<size_t>(num_experts)))
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] route logits scratch allocation failed");
            return false;
        }

        float *legacy_indices = nullptr;
        float *legacy_weights = nullptr;
        if (write_legacy_outputs)
        {
            if (!output_indices || !output_weights)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] legacy outputs requested without tensors");
                return false;
            }

            legacy_indices = static_cast<float *>(output_indices->gpu_data_ptr());
            legacy_weights = static_cast<float *>(output_weights->gpu_data_ptr());
            if (!legacy_indices || !legacy_weights)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] legacy output tensors have no device allocation");
                return false;
            }
        }

        bool logits_ready = false;
        bool runtime_ready = false;
        const auto &rocm_env = debugEnv().rocm;
        const bool gate_is_fp32 = (gate_type == TensorType::FP32);
        invalidateRouterQ8HiddenPublication();
        const bool q8_router_requested = gate_is_fp32 && rocm_env.moe_router_q8;
        // M1 restored-prefix verification uses runtime routing but a grouped
        // shared consumer. Floating router math must still publish that
        // consumer's original-basis Q8 rows on this exact captured stream.
        // The cold demand contract makes ordinary no-consumer decode a no-op.
        if (!q8_router_requested && !publishRequiredRouterQ8Hidden(
                h, /*rows=*/1, d_model, top_k, active_row_count_device))
            return false;
        if (q8_router_requested && (d_model % 32) != 0)
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] Q8 router requires d_model to be a multiple of 32, got "
                      << d_model);
            return false;
        }
        if (q8_router_requested)
        {
            if (!ensureRouterQ8HiddenScratchCapacity(/*rows=*/1, d_model))
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] Q8 router hidden scratch unavailable for d_model="
                          << d_model);
                return false;
            }
            const auto *q8_gate = getOrCreateQ8RouterGateCache(
                static_cast<const float *>(g), d_model, num_experts);
            if (!q8_gate)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] Q8 router gate cache unavailable");
                return false;
            }
            void *stream = requireStream("ROCmMoEKernel::decodeRouteSelect Q8 router");
            if (!stream)
                return false;
            logits_ready = hipMoE_gate_logits_single_token_q8_weights(
                h, d_router_q8_hidden_, d_router_q8_hidden_scales_,
                q8_gate->d_gate_weights_q8, q8_gate->d_gate_scales,
                d_route_logits_, d_model, num_experts,
                device_ordinal_, stream, routerQ8Basis(debugEnv().gemm.moe_router_arithmetic));
            if (!logits_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] Q8 router logits kernel failed");
                return false;
            }
            publishRouterQ8Hidden(
                h,
                /*rows=*/1,
                isDecodeGraphCaptureActive(),
                routerQ8Basis(debugEnv().gemm.moe_router_arithmetic));
        }

        if (!logits_ready && gate_is_fp32 && rocm_env.moe_router_kpart_decode)
        {
            const int k_partitions = rocm_env.moe_router_kparts;
            const size_t partial_count = static_cast<size_t>(num_experts) * static_cast<size_t>(k_partitions);
            if (!ensureRouteLogitsPartialsCapacity(partial_count))
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] K-part router was requested but scratch allocation failed");
                return false;
            }
            const bool partials_ready = hipMoE_gate_logits_single_token_kpart_partials(
                h, static_cast<const float *>(g), d_route_logits_partials_,
                d_model, num_experts, k_partitions,
                device_ordinal_, getStream());
            if (!partials_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] K-part router logits kernel failed");
                return false;
            }
            runtime_ready = hipMoE_router_kpart_reduce_softmax_topk_decode_runtime(
                d_route_logits_partials_,
                d_route_logits_,
                static_cast<void *>(runtime_layer),
                legacy_indices,
                legacy_weights,
                num_experts,
                k_partitions,
                top_k,
                normalize_weights,
                write_legacy_outputs,
                update_runtime_histogram,
                row_execution_policy,
                nullptr,
                nullptr,
                0u,
                nullptr,
                nullptr,
                0u,
                nullptr,
                nullptr,
                nullptr,
                -1,
                1u,
                absolute_position_ids_device,
                device_ordinal_,
                getStream(),
                active_row_count_device);
            if (!runtime_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] fused K-part router runtime kernel failed");
                return false;
            }
        }

        if (!runtime_ready && !logits_ready &&
            !launchDecodeGateLogitsForGateType(
                h, g, gate_type, d_route_logits_,
                d_model, num_experts,
                device_ordinal_, getStream(),
                "ROCmMoEKernel::decodeRouteSelect"))
        {
            return false;
        }

        if (paired_router_descriptors_)
        {
            if (runtime_ready)
                throw std::logic_error("Paired descriptor router selected a different prepared producer");
            runtime_ready = hipMoE_softmax_topk_decode_selected_descriptors(
                d_route_logits_, runtime_layer, legacy_indices, legacy_weights,
                num_experts, top_k, normalize_weights, write_legacy_outputs,
                update_runtime_histogram, row_execution_policy,
                paired_router_descriptors_->outputs, device_ordinal_, getStream(), active_row_count_device);
            if (!runtime_ready)
                throw std::runtime_error("Paired ROCm router/descriptor publication failed");
            PerfStatsCollector::addCounter("moe", "router_selected_descriptors_enqueued", 1.0,
                "graph_capture", DeviceId::rocm(device_ordinal_).toString(),
                {{"descriptor_source", "original_intermediate_shard"}});
        }

        if (!runtime_ready && rocm_env.moe_router_wave_topk &&
            num_experts <= static_cast<int>(kDeviceMoEMaxExperts) &&
            row_execution_policy == RoutedExpertRowExecutionPolicy::ParticipantAssigned &&
            prepared_decode_resident_set_policy_ ==
                RoutedExpertResidentSetPolicy::UniqueOwner)
        {
            runtime_ready =
                hipMoE_softmax_topk_decode_unique_owner_wave64(
                    d_route_logits_,
                    static_cast<void *>(runtime_layer),
                    legacy_indices,
                    legacy_weights,
                    num_experts,
                    top_k,
                    normalize_weights,
                    write_legacy_outputs,
                    update_runtime_histogram,
                    device_ordinal_,
                    getStream(),
                    active_row_count_device);
            if (!runtime_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] unique-owner "
                          "wave64 decode router failed");
                return false;
            }
        }

        if (!runtime_ready && rocm_env.moe_router_wave_topk && num_experts <= static_cast<int>(kDeviceMoEMaxExperts))
        {
            runtime_ready = hipMoE_softmax_topk_decode_runtime_wave64(
                d_route_logits_,
                static_cast<void *>(runtime_layer),
                legacy_indices,
                legacy_weights,
                num_experts,
                top_k,
                normalize_weights,
                write_legacy_outputs,
                update_runtime_histogram,
                row_execution_policy,
                nullptr,
                nullptr,
                0u,
                nullptr,
                nullptr,
                0u,
                nullptr,
                nullptr,
                nullptr,
                -1,
                1u,
                absolute_position_ids_device,
                device_ordinal_,
                getStream(),
                active_row_count_device);
            if (!runtime_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] wave64 decode softmax/top-k runtime kernel failed");
                return false;
            }
        }

        if (!runtime_ready && !hipMoE_softmax_topk_decode_runtime(
                                  d_route_logits_,
                                  static_cast<void *>(runtime_layer),
                                  legacy_indices,
                                  legacy_weights,
                                  num_experts,
                                  top_k,
                                  normalize_weights,
                                  write_legacy_outputs,
                                  update_runtime_histogram,
                                  row_execution_policy,
                                  nullptr,
                                  nullptr,
                                  0u,
                                  nullptr,
                                  nullptr,
                                  0u,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  -1,
                                  1u,
                                  absolute_position_ids_device,
                                  device_ordinal_,
                                  getStream(),
                                  active_row_count_device))
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelect] decode softmax/top-k runtime kernel failed");
            return false;
        }

        if (write_legacy_outputs)
        {
            const DeviceId device = DeviceId::rocm(device_ordinal_);
            markDeviceWritten(output_indices, device, getStream());
            markDeviceWritten(output_weights, device, getStream());
        }

        return true;
    }

    bool ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply(
        DeviceMoELayerRuntime *runtime_layers,
        DeviceMoELayerRuntime *runtime_layer,
        ITensor *hidden, ITensor *gate_weights,
        int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        bool write_legacy_outputs,
        bool update_runtime_histogram,
        const DeviceMoERebalancePlanEntry *rebalance_plan_entries,
        uint32_t rebalance_plan_capacity,
        DeviceMoERebalanceCommandBufferHeader *rebalance_command_header,
        const DeviceMoEExpertDirectoryEntry *rebalance_local_transfer_slots,
        uint32_t rebalance_local_transfer_slot_count,
        const DeviceMoERebalanceConfig &rebalance_config,
        DeviceMoERebalanceApplyStatus *rebalance_apply_status,
        DeviceMoERebalanceGraphControllerState *rebalance_controller_state,
        int rebalance_target_layer,
        uint32_t rebalance_command_buffer_count,
        const int32_t *absolute_position_ids_device,
        RoutedExpertRowExecutionPolicy row_execution_policy,
        const int32_t *active_row_count_device)
    {
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::MOE_ROUTE, static_cast<hipStream_t>(getStream()));

        if (!runtime_layers || !runtime_layer || !hidden || !gate_weights ||
            !rebalance_plan_entries || rebalance_plan_capacity == 0 ||
            !rebalance_command_header || !rebalance_apply_status ||
            !rebalance_controller_state ||
            !validateDeviceMoERebalanceConfig(rebalance_config))
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] invalid runtime or rebalance binding");
            return false;
        }
        if (d_model <= 0 || num_experts <= 0 || top_k <= 0 || top_k > num_experts)
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] invalid dimensions d_model=" << d_model
                                                                                                               << " num_experts=" << num_experts
                                                                                                               << " top_k=" << top_k);
            return false;
        }

        const DeviceId device = DeviceId::rocm(device_ordinal_);
        if (!requireTensorOnDevice(hidden, device, getStream(), "hidden", "decodeRouteSelectWithReadyRebalanceApply") ||
            !requireTensorOnDevice(gate_weights, device, getStream(), "gate_weights", "decodeRouteSelectWithReadyRebalanceApply"))
        {
            return false;
        }

        const float *h = static_cast<const float *>(hidden->gpu_data_ptr());
        const void *g = gate_weights->gpu_data_ptr();
        const TensorType gate_type = gate_weights->native_type();
        if (!h || !g)
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] null device pointer "
                      "(hidden="
                      << (const void *)h << " gate=" << (const void *)g << ")");
            return false;
        }

        if (!ensureRouteBufferCapacity(static_cast<size_t>(num_experts)))
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] route logits scratch allocation failed");
            return false;
        }

        float *legacy_indices = nullptr;
        float *legacy_weights = nullptr;
        if (write_legacy_outputs)
        {
            if (!output_indices || !output_weights ||
                !requireOutputOnDevice(output_indices, device, getStream(), "output_indices", "decodeRouteSelectWithReadyRebalanceApply") ||
                !requireOutputOnDevice(output_weights, device, getStream(), "output_weights", "decodeRouteSelectWithReadyRebalanceApply"))
            {
                return false;
            }

            legacy_indices = static_cast<float *>(output_indices->gpu_data_ptr());
            legacy_weights = static_cast<float *>(output_weights->gpu_data_ptr());
            if (!legacy_indices || !legacy_weights)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] legacy output tensors have no device allocation");
                return false;
            }
        }

        bool logits_ready = false;
        bool runtime_ready = false;
        const auto &rocm_env = debugEnv().rocm;
        const bool gate_is_fp32 = (gate_type == TensorType::FP32);
        invalidateRouterQ8HiddenPublication();
        const bool q8_router_requested = gate_is_fp32 && rocm_env.moe_router_q8;
        // M1 restored-prefix verification uses runtime routing but a grouped
        // shared consumer. Floating router math must still publish that
        // consumer's original-basis Q8 rows on this exact captured stream.
        // The cold demand contract makes ordinary no-consumer decode a no-op.
        if (!q8_router_requested && !publishRequiredRouterQ8Hidden(
                h, /*rows=*/1, d_model, top_k, active_row_count_device))
            return false;
        if (q8_router_requested && (d_model % 32) != 0)
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] Q8 router requires d_model to be a multiple of 32, got "
                      << d_model);
            return false;
        }
        if (q8_router_requested)
        {
            if (!ensureRouterQ8HiddenScratchCapacity(/*rows=*/1, d_model))
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] Q8 router hidden scratch unavailable for d_model="
                          << d_model);
                return false;
            }
            const auto *q8_gate = getOrCreateQ8RouterGateCache(
                static_cast<const float *>(g), d_model, num_experts);
            if (!q8_gate)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] Q8 router gate cache unavailable");
                return false;
            }
            void *stream = requireStream("ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply Q8 router");
            if (!stream)
                return false;
            logits_ready = hipMoE_gate_logits_single_token_q8_weights(
                h, d_router_q8_hidden_, d_router_q8_hidden_scales_,
                q8_gate->d_gate_weights_q8, q8_gate->d_gate_scales,
                d_route_logits_, d_model, num_experts,
                device_ordinal_, stream, routerQ8Basis(debugEnv().gemm.moe_router_arithmetic));
            if (!logits_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] Q8 router logits kernel failed");
                return false;
            }
            publishRouterQ8Hidden(
                h,
                /*rows=*/1,
                isDecodeGraphCaptureActive(),
                routerQ8Basis(debugEnv().gemm.moe_router_arithmetic));
        }

        if (!logits_ready && gate_is_fp32 && rocm_env.moe_router_kpart_decode)
        {
            const int k_partitions = rocm_env.moe_router_kparts;
            const size_t partial_count = static_cast<size_t>(num_experts) * static_cast<size_t>(k_partitions);
            if (!ensureRouteLogitsPartialsCapacity(partial_count))
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] K-part router was requested but scratch allocation failed");
                return false;
            }
            const bool partials_ready = hipMoE_gate_logits_single_token_kpart_partials(
                h, static_cast<const float *>(g), d_route_logits_partials_,
                d_model, num_experts, k_partitions,
                device_ordinal_, getStream());
            if (!partials_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] K-part router logits kernel failed");
                return false;
            }
            runtime_ready = hipMoE_router_kpart_reduce_softmax_topk_decode_runtime(
                d_route_logits_partials_,
                d_route_logits_,
                static_cast<void *>(runtime_layer),
                legacy_indices,
                legacy_weights,
                num_experts,
                k_partitions,
                top_k,
                normalize_weights,
                write_legacy_outputs,
                update_runtime_histogram,
                row_execution_policy,
                runtime_layers,
                rebalance_plan_entries,
                rebalance_plan_capacity,
                rebalance_command_header,
                rebalance_local_transfer_slots,
                rebalance_local_transfer_slot_count,
                &rebalance_config,
                rebalance_apply_status,
                rebalance_controller_state,
                rebalance_target_layer,
                rebalance_command_buffer_count,
                absolute_position_ids_device,
                device_ordinal_,
                getStream(),
                active_row_count_device);
            if (!runtime_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] fused K-part router runtime kernel failed");
                return false;
            }
        }

        if (!runtime_ready && !logits_ready &&
            !launchDecodeGateLogitsForGateType(
                h, g, gate_type, d_route_logits_,
                d_model, num_experts,
                device_ordinal_, getStream(),
                "ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply"))
        {
            return false;
        }

        if (!runtime_ready && rocm_env.moe_router_wave_topk && num_experts <= static_cast<int>(kDeviceMoEMaxExperts))
        {
            runtime_ready = hipMoE_softmax_topk_decode_runtime_wave64(
                d_route_logits_,
                static_cast<void *>(runtime_layer),
                legacy_indices,
                legacy_weights,
                num_experts,
                top_k,
                normalize_weights,
                write_legacy_outputs,
                update_runtime_histogram,
                row_execution_policy,
                runtime_layers,
                rebalance_plan_entries,
                rebalance_plan_capacity,
                rebalance_command_header,
                rebalance_local_transfer_slots,
                rebalance_local_transfer_slot_count,
                &rebalance_config,
                rebalance_apply_status,
                rebalance_controller_state,
                rebalance_target_layer,
                rebalance_command_buffer_count,
                absolute_position_ids_device,
                device_ordinal_,
                getStream(),
                active_row_count_device);
            if (!runtime_ready)
            {
                LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] wave64 decode softmax/top-k runtime kernel failed");
                return false;
            }
        }

        if (!runtime_ready && !hipMoE_softmax_topk_decode_runtime(
                                  d_route_logits_,
                                  static_cast<void *>(runtime_layer),
                                  legacy_indices,
                                  legacy_weights,
                                  num_experts,
                                  top_k,
                                  normalize_weights,
                                  write_legacy_outputs,
                                  update_runtime_histogram,
                                  row_execution_policy,
                                  runtime_layers,
                                  rebalance_plan_entries,
                                  rebalance_plan_capacity,
                                  rebalance_command_header,
                                  rebalance_local_transfer_slots,
                                  rebalance_local_transfer_slot_count,
                                  &rebalance_config,
                                  rebalance_apply_status,
                                  rebalance_controller_state,
                                  rebalance_target_layer,
                                  rebalance_command_buffer_count,
                                  absolute_position_ids_device,
                                  device_ordinal_,
                                  getStream(),
                                  active_row_count_device))
        {
            LOG_ERROR("[ROCmMoEKernel::decodeRouteSelectWithReadyRebalanceApply] decode softmax/top-k runtime kernel failed");
            return false;
        }

        if (write_legacy_outputs)
        {
            markDeviceWritten(output_indices, device, getStream());
            markDeviceWritten(output_weights, device, getStream());
        }

        return true;
    }

    bool ROCmMoEKernel::runDeviceRebalanceController(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        const uint64_t *gathered_histograms,
        DeviceMoERebalanceStatus *status,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalancePlanEntry *plan_entries,
        uint32_t *plan_count,
        uint32_t plan_capacity,
        uint32_t payload_slot_capacity,
        DeviceMoERebalanceCommandBufferHeader *command_header,
        DeviceMoERebalanceWaveState *wave_state,
        DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count,
        const DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        DeviceMoELLEPLayerPlanScratch *llep_layer_plans,
        DeviceMoEPlacementBank *placement_plan_scratch)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::runDeviceRebalanceController] invalid device rebalance config");
            return false;
        }
        const bool deferred_planning = hasDeviceMoERebalanceFlag(
            config.flags,
            DeviceMoERebalanceFlags::DeferRuntimeApply);
        if (!runtime_layers || !gathered_histograms || !status ||
            !controller_state ||
            (deferred_planning && !placement_plan_scratch))
        {
            LOG_ERROR("[ROCmMoEKernel::runDeviceRebalanceController] runtime layers, gathered histograms, status, controller state, and placement planner scratch must be non-null");
            return false;
        }
        if (config.routed_assignment_policy ==
                kDeviceMoERebalanceAssignmentLeastLoadedResident &&
            !llep_layer_plans)
        {
            LOG_ERROR("[ROCmMoEKernel::runDeviceRebalanceController] LLEP requires graph-lifetime parallel planner scratch");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "runDeviceRebalanceController");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "runDeviceRebalanceController"))
            return false;

        return hipMoE_device_rebalance_controller(
            runtime_layers,
            reinterpret_cast<const unsigned long long *>(gathered_histograms),
            status,
            &config,
            plan_entries,
            plan_count,
            plan_capacity,
            payload_slot_capacity,
            command_header,
            wave_state,
            controller_state,
            llep_layer_plans,
            placement_plan_scratch,
            command_buffer_count,
            local_transfer_slots,
            local_transfer_slot_count,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::packDeviceRebalanceHistograms(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        uint64_t *local_histograms,
        const DeviceMoERebalanceConfig &config,
        const DeviceMoERebalanceWaveState *wave_state,
        const DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count,
        uint32_t histogram_source_mask,
        uint64_t *previous_activation_counts)
    {
        if (!validateDeviceMoERebalanceConfig(config) ||
            !moe_runtime_abi::validHistogramSourceMask(
                histogram_source_mask))
        {
            LOG_ERROR("[ROCmMoEKernel::packDeviceRebalanceHistograms] invalid device rebalance config or histogram source mask");
            return false;
        }
        if (!runtime_layers || !local_histograms)
        {
            LOG_ERROR("[ROCmMoEKernel::packDeviceRebalanceHistograms] runtime layers and local histograms must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "packDeviceRebalanceHistograms");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "packDeviceRebalanceHistograms"))
            return false;

        return hipMoE_pack_rebalance_histograms(
            runtime_layers,
            reinterpret_cast<unsigned long long *>(local_histograms),
            &config,
            wave_state,
            controller_state,
            command_buffer_count,
            histogram_source_mask,
            reinterpret_cast<unsigned long long *>(
                previous_activation_counts),
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::packDeviceRebalanceDirectory(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        DeviceMoEExpertDirectoryEntry *local_directory,
        const DeviceMoERebalanceConfig &config)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::packDeviceRebalanceDirectory] invalid device rebalance config");
            return false;
        }
        if (!runtime_layers || !local_directory)
        {
            LOG_ERROR("[ROCmMoEKernel::packDeviceRebalanceDirectory] runtime layers and local directory must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "packDeviceRebalanceDirectory");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "packDeviceRebalanceDirectory"))
            return false;

        return hipMoE_pack_rebalance_directory(
            runtime_layers,
            local_directory,
            &config,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::packDeviceRebalanceSourceDescriptors(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        const DeviceMoERebalancePlanEntry *plan_entries,
        const DeviceMoERebalanceCommandBufferHeader *command_headers,
        uint32_t plan_capacity,
        DeviceMoEExpertDirectoryEntry *local_source_descriptors,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::packDeviceRebalanceSourceDescriptors] invalid device rebalance config");
            return false;
        }
        if (!runtime_layers || !plan_entries || !command_headers ||
            !local_source_descriptors)
        {
            LOG_ERROR("[ROCmMoEKernel::packDeviceRebalanceSourceDescriptors] runtime, projected plans, headers, and source descriptor output must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "packDeviceRebalanceSourceDescriptors");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "packDeviceRebalanceSourceDescriptors"))
            return false;

        return hipMoE_pack_rebalance_source_descriptors(
            runtime_layers,
            plan_entries,
            command_headers,
            plan_capacity,
            local_source_descriptors,
            &config,
            controller_state,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::projectDeviceRebalanceDomainCommands(
        const MoEKernelLaunchContext &launch,
        const DeviceMoERebalancePlanEntry *gathered_plan_entries,
        const DeviceMoERebalanceCommandBufferHeader *gathered_command_headers,
        uint32_t plan_capacity,
        DeviceMoERebalancePlanEntry *local_plan_entries,
        DeviceMoERebalanceCommandBufferHeader *local_command_headers,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceStatus *status,
        uint32_t payload_slot_capacity,
        uint32_t command_buffer_count,
        const DeviceMoERebalanceWaveState *gathered_wave_states,
        DeviceMoERebalanceWaveState *local_wave_states,
        DeviceMoELayerRuntime *runtime_layers,
        const DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        DeviceMoETransferSlotClaimIndex *transfer_slot_claim_index)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::projectDeviceRebalanceDomainCommands] invalid device rebalance config");
            return false;
        }
        if (!gathered_plan_entries || !gathered_command_headers ||
            !local_plan_entries || !local_command_headers ||
            !runtime_layers || !local_transfer_slots ||
            !transfer_slot_claim_index ||
            plan_capacity == 0 || local_transfer_slot_count == 0)
        {
            LOG_ERROR("[ROCmMoEKernel::projectDeviceRebalanceDomainCommands] gathered/local commands, runtime layers, and transfer directory must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "projectDeviceRebalanceDomainCommands");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "projectDeviceRebalanceDomainCommands"))
            return false;

        return hipMoE_project_rebalance_domain_commands(
            gathered_plan_entries,
            gathered_command_headers,
            plan_capacity,
            local_plan_entries,
            local_command_headers,
            &config,
            status,
            payload_slot_capacity,
            command_buffer_count,
            gathered_wave_states,
            local_wave_states,
            runtime_layers,
            local_transfer_slots,
            local_transfer_slot_count,
            transfer_slot_claim_index,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::projectPrefillLeastLoadedDomainCommands(
        const MoEKernelLaunchContext &launch,
        const DeviceMoERebalancePlanEntry *gathered_plan_entries,
        const DeviceMoERebalanceCommandBufferHeader *gathered_command_headers,
        uint32_t plan_capacity,
        DeviceMoERebalancePlanEntry *local_plan_entries,
        uint32_t *local_plan_count,
        DeviceMoERebalanceCommandBufferHeader *local_command_header,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceStatus *status,
        uint32_t payload_slot_capacity,
        DeviceMoELayerRuntime *runtime_layers,
        const DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        DeviceMoETransferSlotClaimIndex *transfer_slot_claim_index,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::projectPrefillLeastLoadedDomainCommands] invalid device rebalance config");
            return false;
        }
        if (!gathered_plan_entries || !gathered_command_headers ||
            !local_plan_entries || !local_plan_count || !local_command_header ||
            !status || !runtime_layers || !local_transfer_slots ||
            !transfer_slot_claim_index ||
            plan_capacity == 0 || payload_slot_capacity == 0 ||
            local_transfer_slot_count == 0)
        {
            LOG_ERROR("[ROCmMoEKernel::projectPrefillLeastLoadedDomainCommands] gathered commands, local command output, runtime layers, transfer directory, status, and payload capacity are required");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "projectPrefillLeastLoadedDomainCommands");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "projectPrefillLeastLoadedDomainCommands"))
            return false;

        return hipMoE_project_prefill_llep_domain_commands(
            gathered_plan_entries,
            gathered_command_headers,
            plan_capacity,
            local_plan_entries,
            local_plan_count,
            local_command_header,
            &config,
            status,
            payload_slot_capacity,
            command_buffer_count,
            runtime_layers,
            local_transfer_slots,
            local_transfer_slot_count,
            transfer_slot_claim_index,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::materializePrefillLeastLoadedMirroredDomainCommands(
        const MoEKernelLaunchContext &launch,
        const DeviceMoELayerRuntime *runtime_layer,
        DeviceMoERebalancePlanEntry *mirrored_plan_entries,
        DeviceMoERebalanceCommandBufferHeader *mirrored_command_headers,
        uint32_t plan_capacity,
        DeviceMoERebalanceStatus *status,
        const DeviceMoERebalanceConfig &config,
        uint32_t payload_slot_capacity,
        uint32_t layer_idx)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::materializePrefillLeastLoadedMirroredDomainCommands] invalid device rebalance config");
            return false;
        }
        if (!runtime_layer || !mirrored_plan_entries ||
            !mirrored_command_headers || !status || plan_capacity == 0 ||
            payload_slot_capacity == 0)
        {
            LOG_ERROR("[ROCmMoEKernel::materializePrefillLeastLoadedMirroredDomainCommands] runtime, mirrored command buffers, status, and payload capacity are required");
            return false;
        }
        if (layer_idx >= config.num_layers)
        {
            LOG_ERROR("[ROCmMoEKernel::materializePrefillLeastLoadedMirroredDomainCommands] layer index out of range"
                      << " layer=" << layer_idx
                      << " num_layers=" << config.num_layers);
            return false;
        }
        void *stream = explicitMoELaunchStream(
            launch,
            "materializePrefillLeastLoadedMirroredDomainCommands");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(
                device_ordinal_,
                "materializePrefillLeastLoadedMirroredDomainCommands"))
        {
            return false;
        }

        return hipMoE_materialize_prefill_llep_mirrored_domain_commands(
            runtime_layer,
            mirrored_plan_entries,
            mirrored_command_headers,
            plan_capacity,
            status,
            &config,
            payload_slot_capacity,
            layer_idx,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::materializePrefillLeastLoadedTransferCommands(
        const MoEKernelLaunchContext &launch,
        const DeviceMoELayerRuntime *runtime_layer,
        DeviceMoERebalancePlanEntry *plan_entries,
        uint32_t *plan_count,
        uint32_t plan_capacity,
        DeviceMoERebalanceCommandBufferHeader *command_header,
        DeviceMoERebalanceStatus *status,
        const DeviceMoERebalanceConfig &config,
        uint32_t payload_slot_capacity,
        uint32_t layer_idx,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::materializePrefillLeastLoadedTransferCommands] invalid device rebalance config");
            return false;
        }
        if (!runtime_layer || !plan_entries || !plan_count || !command_header || !status ||
            plan_capacity == 0 || payload_slot_capacity == 0)
        {
            LOG_ERROR("[ROCmMoEKernel::materializePrefillLeastLoadedTransferCommands] runtime, command buffers, status, and payload capacity are required");
            return false;
        }
        if (layer_idx >= config.num_layers)
        {
            LOG_ERROR("[ROCmMoEKernel::materializePrefillLeastLoadedTransferCommands] layer index out of range"
                      << " layer=" << layer_idx << " num_layers=" << config.num_layers);
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "materializePrefillLeastLoadedTransferCommands");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "materializePrefillLeastLoadedTransferCommands"))
            return false;

        return hipMoE_materialize_prefill_llep_transfer_commands(
            runtime_layer,
            plan_entries,
            plan_count,
            plan_capacity,
            command_header,
            status,
            &config,
            payload_slot_capacity,
            layer_idx,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::packDeviceRebalanceCompactPayloads(
        const MoEKernelLaunchContext &launch,
        const DeviceMoERebalancePlanEntry *plan_entries,
        const DeviceMoERebalanceCommandBufferHeader *command_headers,
        uint32_t plan_capacity,
        const DeviceMoEExpertDirectoryEntry *local_source_descriptors,
        uint8_t *local_payload,
        uint32_t local_payload_slot_count,
        uint64_t payload_slot_bytes,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status,
        DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::packDeviceRebalanceCompactPayloads] invalid device rebalance config");
            return false;
        }
        if (!plan_entries || !command_headers ||
            !local_source_descriptors || !local_payload || !status ||
            plan_capacity == 0 || local_payload_slot_count == 0 ||
            payload_slot_bytes <= sizeof(DeviceMoEExpertDirectoryEntry))
        {
            LOG_ERROR("[ROCmMoEKernel::packDeviceRebalanceCompactPayloads] plans, headers, source descriptors, payload, and status must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "packDeviceRebalanceCompactPayloads");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "packDeviceRebalanceCompactPayloads"))
            return false;

        return hipMoE_pack_rebalance_compact_payloads(
            plan_entries,
            command_headers,
            plan_capacity,
            local_source_descriptors,
            local_payload,
            local_payload_slot_count,
            static_cast<unsigned long long>(payload_slot_bytes),
            &config,
            status,
            controller_state,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::applyDeviceRebalanceArrivals(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layers,
        const DeviceMoERebalancePlanEntry *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status,
        const DeviceMoERebalanceCommandBufferHeader *command_header,
        int target_layer)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::applyDeviceRebalanceArrivals] invalid device rebalance config");
            return false;
        }
        if (!runtime_layers || !plan_entries || (!plan_count && !command_header) ||
            !local_transfer_slots || !status)
        {
            LOG_ERROR("[ROCmMoEKernel::applyDeviceRebalanceArrivals] runtime, plan, transfer slots, and status must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "applyDeviceRebalanceArrivals");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "applyDeviceRebalanceArrivals"))
            return false;

        return hipMoE_apply_rebalance_arrivals(
            runtime_layers,
            plan_entries,
            plan_count,
            plan_capacity,
            command_header,
            local_transfer_slots,
            local_transfer_slot_count,
            &config,
            status,
            target_layer,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::packDeviceRebalanceCollectivePayloads(
        const MoEKernelLaunchContext &launch,
        const DeviceMoERebalancePlanEntry *gathered_plan_entries,
        const DeviceMoERebalanceCommandBufferHeader *gathered_command_headers,
        uint32_t plan_capacity,
        const DeviceMoEExpertDirectoryEntry *local_directory,
        uint8_t *local_payload,
        uint32_t local_payload_slot_count,
        uint64_t payload_slot_bytes,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status,
        DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::packDeviceRebalanceCollectivePayloads] invalid device rebalance config");
            return false;
        }
        if (!gathered_plan_entries || !gathered_command_headers ||
            !local_directory || !local_payload || !status ||
            plan_capacity == 0 || local_payload_slot_count == 0 ||
            payload_slot_bytes == 0)
        {
            LOG_ERROR("[ROCmMoEKernel::packDeviceRebalanceCollectivePayloads] gathered plans, headers, local directory, payload, and status must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "packDeviceRebalanceCollectivePayloads");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "packDeviceRebalanceCollectivePayloads"))
            return false;

        return hipMoE_pack_rebalance_collective_payloads(
            gathered_plan_entries,
            gathered_command_headers,
            plan_capacity,
            local_directory,
            local_payload,
            local_payload_slot_count,
            static_cast<unsigned long long>(payload_slot_bytes),
            &config,
            status,
            controller_state,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::unpackDeviceRebalanceCollectivePayloads(
        const MoEKernelLaunchContext &launch,
        const DeviceMoERebalancePlanEntry *plan_entries,
        const uint32_t *plan_count,
        uint32_t plan_capacity,
        const DeviceMoERebalanceCommandBufferHeader *command_header,
        const uint8_t *gathered_payload,
        uint32_t local_payload_slot_count,
        uint64_t payload_slot_bytes,
        DeviceMoEExpertDirectoryEntry *local_transfer_slots,
        uint32_t local_transfer_slot_count,
        const DeviceMoERebalanceConfig &config,
        DeviceMoERebalanceApplyStatus *status,
        DeviceMoERebalanceGraphControllerState *controller_state,
        uint32_t command_buffer_count)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::unpackDeviceRebalanceCollectivePayloads] invalid device rebalance config");
            return false;
        }
        if (!plan_entries || (!plan_count && !command_header) ||
            !gathered_payload || !local_transfer_slots || !status ||
            plan_capacity == 0 || local_payload_slot_count == 0 ||
            payload_slot_bytes <= sizeof(DeviceMoEExpertDirectoryEntry))
        {
            LOG_ERROR("[ROCmMoEKernel::unpackDeviceRebalanceCollectivePayloads] plan, gathered payload, transfer slots, and status must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "unpackDeviceRebalanceCollectivePayloads");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "unpackDeviceRebalanceCollectivePayloads"))
            return false;

        return hipMoE_unpack_rebalance_collective_payloads(
            plan_entries,
            plan_count,
            plan_capacity,
            command_header,
            gathered_payload,
            local_payload_slot_count,
            static_cast<unsigned long long>(payload_slot_bytes),
            local_transfer_slots,
            local_transfer_slot_count,
            &config,
            status,
            controller_state,
            command_buffer_count,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::initializeDeviceRebalanceGraphController(
        const MoEKernelLaunchContext &launch,
        DeviceMoERebalanceGraphControllerState *controller_state,
        const DeviceMoERebalanceConfig &config)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::initializeDeviceRebalanceGraphController] invalid device rebalance config");
            return false;
        }
        if (!controller_state)
        {
            LOG_ERROR("[ROCmMoEKernel::initializeDeviceRebalanceGraphController] controller state must be non-null");
            return false;
        }
        void *stream = explicitMoELaunchStream(launch, "initializeDeviceRebalanceGraphController");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(device_ordinal_, "initializeDeviceRebalanceGraphController"))
            return false;

        return hipMoE_init_rebalance_graph_controller_state(
            controller_state,
            &config,
            device_ordinal_,
            stream);
    }

    bool ROCmMoEKernel::resetDeviceRebalanceGraphTransactionForRequest(
        const MoEKernelLaunchContext &launch,
        DeviceMoERebalanceGraphControllerState *controller_state,
        DeviceMoERebalanceCommandBufferHeader *command_headers,
        DeviceMoERebalanceWaveState *wave_states,
        uint32_t *plan_counts,
        uint32_t command_buffer_count,
        const DeviceMoERebalanceConfig &config)
    {
        if (!validateDeviceMoERebalanceConfig(config))
        {
            LOG_ERROR("[ROCmMoEKernel::resetDeviceRebalanceGraphTransactionForRequest] invalid device rebalance config");
            return false;
        }
        if (!controller_state || !command_headers || !wave_states ||
            !plan_counts || command_buffer_count == 0u ||
            command_buffer_count > 2u)
        {
            LOG_ERROR("[ROCmMoEKernel::resetDeviceRebalanceGraphTransactionForRequest] transaction buffers must be non-null and command-buffer count must be one or two");
            return false;
        }
        void *stream = explicitMoELaunchStream(
            launch,
            "resetDeviceRebalanceGraphTransactionForRequest");
        if (!stream)
            return false;
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(
            ROCmKernelType::MOE_ROUTE,
            static_cast<hipStream_t>(stream));
        if (!setMoEDevice(
                device_ordinal_,
                "resetDeviceRebalanceGraphTransactionForRequest"))
        {
            return false;
        }

        return hipMoE_reset_rebalance_graph_transaction_for_request(
            controller_state,
            command_headers,
            wave_states,
            plan_counts,
            command_buffer_count,
            &config,
            device_ordinal_,
            stream);
    }
} // namespace llaminar2
