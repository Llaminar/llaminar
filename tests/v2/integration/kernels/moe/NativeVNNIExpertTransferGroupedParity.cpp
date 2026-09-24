/**
 * @file NativeVNNIExpertTransferGroupedParity.cpp
 * @brief Shared all-codebook compact-transfer and grouped execution proof.
 *
 * Both accelerator fixtures invoke this one implementation. Keeping its large
 * format/M sweep out of registration headers makes a lifecycle fix compile
 * once instead of recompiling both monolithic backend test translation units.
 */
#include "NativeVNNIExpertTransferParityTest.h"

namespace llaminar2::test
{
    /** @copydoc runNativeVNNIExpertTransferGroupedParity */
    void runNativeVNNIExpertTransferGroupedParity(
        const char *backend_label,
        DeviceId device,
        void *stream,
        NativeExpertTransferProofScope scope)
    {
        using namespace native_vnni_transfer_parity_detail;

        if (!backend_label || !stream || !device.is_gpu() ||
            (scope != NativeExpertTransferProofScope::CopyPublication &&
             scope != NativeExpertTransferProofScope::GroupedArithmetic))
            throw std::invalid_argument(
                "NativeVNNI transfer parity requires a GPU and explicit stream");
        IBackend *backend = getBackendFor(device);
        if (!backend)
            throw std::runtime_error("NativeVNNI transfer parity backend is unavailable");

        constexpr int kDModel = 2048;
        constexpr int kIntermediate = 512;
        constexpr int kMaxRows = 2560;
        constexpr int kNumExperts = 256;
        constexpr int kTopK = 8;
        constexpr uint32_t kPlanCapacity = 32u;
        /*
         * Collective payload lanes and physical transfer-directory slots are
         * different address spaces.  Keep the wire transaction compact while
         * forcing both arrivals beyond its width: production long-context LLEP
         * first diverged when layer 1 leased directory slots 32 through 35
         * after layer 0 had retained the lower slots.
         */
        constexpr uint32_t kPhysicalTransferSlotBase = kPlanCapacity;
        constexpr uint32_t kTransferSlotCount =
            kPhysicalTransferSlotBase + kPlanCapacity;
        constexpr uint32_t kCommandBufferCount = 1u;
        constexpr uint32_t kParticipantCount = 2u;
        constexpr uint32_t kDestinationParticipant = 1u;
        constexpr int kRuntimeLayerCount = 2;
        constexpr int kTargetLayer = 1;
        std::array<uint32_t, kPlanCapacity> transferred_experts{};
        for (uint32_t index = 0u; index < kPlanCapacity; ++index)
            transferred_experts[index] = index * 2u + 1u;
        /*
         * Cover the complete MTP regime plus both scalable-grouping boundaries
         * and the production long-context capture bucket that exposed the E2E
         * failure. The largest case is intentionally retained for every format:
         * small-M exactness cannot certify 32-bit count/offset arithmetic.
         */
        constexpr std::array<int, 8> kRows = {
            2, 4, 8, 15, 16, 65, 257, 2560};

        const auto &formats = quantizedMoEVerifierFormats();
        std::vector<std::vector<DeviceMoETransferSlotDirectory::ProjectionSpec>>
            format_specs;
        format_specs.reserve(formats.size());
        for (const auto &format : formats)
        {
            format_specs.push_back(
                projectionSpecsForFormat(format, kDModel, kIntermediate));
        }

        auto transfer_directory = DeviceMoETransferSlotDirectory::createForTest(
            backend,
            device,
            device.ordinal,
            kDestinationParticipant,
            /*slot_count=*/kTransferSlotCount,
            DeviceMoETransferSlotDirectory::profileForLayerFormats(format_specs));
        ASSERT_NE(transfer_directory, nullptr);
        ASSERT_EQ(transfer_directory->slotCount(), kTransferSlotCount);

        auto kernel_owner =
            llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
        ASSERT_NE(kernel_owner, nullptr);
        kernel_owner->setGPUStream(stream);

        auto requirements = device.is_cuda()
                                ? MoEWorkspaceBuffers::cudaMoE(
                                      kMaxRows,
                                      kDModel,
                                      kIntermediate,
                                      kNumExperts,
                                      kTopK)
                                : MoEWorkspaceBuffers::rocmMoE(
                                      kMaxRows,
                                      kDModel,
                                      kIntermediate,
                                      kNumExperts,
                                      kTopK);
        DeviceWorkspaceManager workspace(
            device,
            requirements.total_bytes_with_alignment() + 4u * 1024u * 1024u);
        ASSERT_TRUE(workspace.allocate(requirements));
        auto *workspace_consumer =
            dynamic_cast<IWorkspaceConsumer *>(kernel_owner.get());
        ASSERT_NE(workspace_consumer, nullptr);
        workspace_consumer->bindWorkspace(&workspace);

        IMoEKernel &kernel = *kernel_owner;
        const MoEKernelLaunchContext launch{
            .stream = stream,
            .workspace = &workspace,
        };

        const uint64_t payload_slot_bytes =
            (sizeof(DeviceMoEExpertDirectoryEntry) +
             transfer_directory->wirePayloadBytes() + 255u) /
            256u * 256u;
        const size_t local_payload_bytes =
            kPlanCapacity * static_cast<size_t>(payload_slot_bytes);
        const size_t gathered_payload_bytes =
            kParticipantCount * local_payload_bytes;
        const size_t source_directory_entries =
            kParticipantCount * kCommandBufferCount * kPlanCapacity;

        DeviceAllocation d_plan(
            backend,
            device.ordinal,
            kPlanCapacity * sizeof(DeviceMoERebalancePlanEntry));
        DeviceAllocation d_plan_count(
            backend,
            device.ordinal,
            sizeof(uint32_t));
        DeviceAllocation d_command_header(
            backend,
            device.ordinal,
            sizeof(DeviceMoERebalanceCommandBufferHeader));
        DeviceAllocation d_source_descriptors(
            backend,
            device.ordinal,
            source_directory_entries * sizeof(DeviceMoEExpertDirectoryEntry));
        DeviceAllocation d_local_payload(
            backend,
            device.ordinal,
            local_payload_bytes);
        DeviceAllocation d_gathered_payload(
            backend,
            device.ordinal,
            gathered_payload_bytes);
        DeviceAllocation d_copy_status(
            backend,
            device.ordinal,
            sizeof(DeviceMoERebalanceApplyStatus));
        DeviceAllocation d_apply_status(
            backend,
            device.ordinal,
            sizeof(DeviceMoERebalanceApplyStatus));
        DeviceAllocation d_source_apply_status(
            backend,
            device.ordinal,
            sizeof(DeviceMoERebalanceApplyStatus));
        DeviceAllocation d_transfer_status(
            backend,
            device.ordinal,
            sizeof(DeviceMoERebalanceStatus));

        for (size_t format_index = 0;
             format_index < formats.size();
             ++format_index)
        {
            const auto &format = formats[format_index];
            SCOPED_TRACE(
                std::string(backend_label) + " format=" + format.label);

            std::vector<std::unique_ptr<TensorBase>> weights;
            std::vector<GpuPreparedGemm> prepared;
            weights.reserve(3u);
            prepared.reserve(3u);

            const uint64_t model_base =
                1900000u + static_cast<uint64_t>(format_index) * 64u +
                (device.is_cuda() ? 0u : 100000u);
            const std::string name_prefix =
                std::string("test.") + backend_label +
                ".native_vnni_transfer_parity." + format.label;

            std::vector<DeviceMoEExpertDescriptor> source_descriptors(kNumExperts);
            std::array<DeviceNativeVNNIMatrixDesc, kNumExperts> gate_descs{};
            std::array<DeviceNativeVNNIMatrixDesc, kNumExperts> up_descs{};
            std::array<DeviceNativeVNNIMatrixDesc, kNumExperts> down_descs{};

            /*
             * Cardinality, not unique payload bytes, is the subject of this
             * regression.  Prepare one real production-ABI matrix triplet per
             * format and alias it across 256 logical experts.  The grouped
             * kernels still execute every routed row and every expert group,
             * while setup remains light enough to sweep all codebooks and M.
             */
            const DeviceNativeVNNIMatrixDesc shared_gate =
                prepareMatrixDescriptor(
                    format,
                    device,
                    stream,
                    kIntermediate,
                    kDModel,
                    810001u,
                    name_prefix + ".shared.gate",
                    ModelContextId{model_base + 1u},
                    weights,
                    prepared);
            const DeviceNativeVNNIMatrixDesc shared_up =
                prepareMatrixDescriptor(
                    format,
                    device,
                    stream,
                    kIntermediate,
                    kDModel,
                    810002u,
                    name_prefix + ".shared.up",
                    ModelContextId{model_base + 2u},
                    weights,
                    prepared);
            const DeviceNativeVNNIMatrixDesc shared_down =
                prepareMatrixDescriptor(
                    format,
                    device,
                    stream,
                    kDModel,
                    kIntermediate,
                    810003u,
                    name_prefix + ".shared.down",
                    ModelContextId{model_base + 3u},
                    weights,
                    prepared);
            for (int expert = 0; expert < kNumExperts; ++expert)
            {
                auto &descriptor = source_descriptors[expert];
                descriptor.logical_expert_id = expert;
                descriptor.owner_participant = 0;
                descriptor.local_slot = expert;
                descriptor.flags = toMoEExpertFlags(
                    DeviceMoEExpertFlags::Valid |
                    DeviceMoEExpertFlags::Resident |
                    DeviceMoEExpertFlags::LocalCompute);

                descriptor.gate = shared_gate;
                descriptor.up = shared_up;
                descriptor.down = shared_down;
                gate_descs[expert] = descriptor.gate;
                up_descs[expert] = descriptor.up;
                down_descs[expert] = descriptor.down;
            }

            const int source_gateup_table =
                kernel.uploadGroupedExpertGateUpDescriptorTables(
                    gate_descs.data(),
                    up_descs.data(),
                    kNumExperts,
                    kDModel,
                    kIntermediate);
            const int source_down_table =
                kernel.uploadGroupedExpertDownDescriptorTable(
                    down_descs.data(),
                    kNumExperts,
                    kDModel,
                    kIntermediate);
            const int destination_gateup_table =
                kernel.uploadGroupedExpertGateUpDescriptorTables(
                    gate_descs.data(),
                    up_descs.data(),
                    kNumExperts,
                    kDModel,
                    kIntermediate);
            const int destination_down_table =
                kernel.uploadGroupedExpertDownDescriptorTable(
                    down_descs.data(),
                    kNumExperts,
                    kDModel,
                    kIntermediate);
            ASSERT_GE(source_gateup_table, 0);
            ASSERT_GE(source_down_table, 0);
            ASSERT_GE(destination_gateup_table, 0);
            ASSERT_GE(destination_down_table, 0);

            // A current-batch arrival is a transient child of an acquired
            // durable overlay epoch. Give each simulated participant its own
            // production arena and hold the reader ticket through both layers
            // and every grouped execution. A bare runtime table cannot certify
            // this protocol even when the payload copy itself is correct.
            auto source_epoch = std::make_shared<DeviceMoEOverlayEpochArena>(
                DeviceMoEOverlayEpochArena::Config{
                    .device_id = device, .initial_epoch = 1u,
                    .initial_bank = 1u, .request_slot_capacity = 1u});
            auto destination_epoch = std::make_shared<DeviceMoEOverlayEpochArena>(
                DeviceMoEOverlayEpochArena::Config{
                    .device_id = device, .initial_epoch = 1u,
                    .initial_bank = 1u, .request_slot_capacity = 1u});
            auto source_placement = makeRuntimeTable(
                device,
                stream,
                source_descriptors,
                /*participant_id=*/0u,
                std::vector<uint8_t>(kNumExperts, 1u),
                std::vector<uint32_t>(kNumExperts, 0b01u),
                kRuntimeLayerCount,
                kTopK,
                kMaxRows,
                source_epoch);
            std::vector<uint8_t> destination_compute_mask(kNumExperts, 1u);
            std::vector<uint32_t> destination_resident_mask(kNumExperts, 0b11u);
            for (const uint32_t expert : transferred_experts)
            {
                destination_compute_mask[expert] = 0u;
                destination_resident_mask[expert] = 0b01u;
            }
            auto destination_placement = makeRuntimeTable(
                device,
                stream,
                source_descriptors,
                kDestinationParticipant,
                destination_compute_mask,
                destination_resident_mask,
                kRuntimeLayerCount,
                kTopK,
                kMaxRows,
                destination_epoch);

            // Current-batch LLEP publishes only into transient child banks;
            // the canonical placement and its acquired epoch remain immutable.
            // Use the production child-table binding, not a manually installed
            // bank pointer or a forged ticket in an otherwise bare table.
            auto source_runtime = makeRuntimeTable(
                device, stream, source_descriptors, 0u,
                std::vector<uint8_t>(kNumExperts, 1u),
                std::vector<uint32_t>(kNumExperts, 0b01u),
                kRuntimeLayerCount, kTopK, kMaxRows, source_epoch,
                source_placement.get());
            auto destination_runtime = makeRuntimeTable(
                device, stream, source_descriptors, kDestinationParticipant,
                destination_compute_mask, destination_resident_mask,
                kRuntimeLayerCount, kTopK, kMaxRows, destination_epoch,
                destination_placement.get());

            for (const auto &epoch : {source_epoch, destination_epoch})
                ASSERT_TRUE(kernel.acquireMoEOverlayEpoch(
                    launch, epoch->control(), epoch->requestTicket(0u),
                    epoch->requestStatus(0u)));

            /*
             * The host runtime object owns only immutable scratch addresses in
             * this setup. Publish the two current-batch counts once before GPU
             * work begins; all route values, placement mutation, and grouped
             * execution remain device-owned after this boundary.
             */
            auto &source_host = source_runtime->hostLayerState(kTargetLayer);
            auto &destination_host =
                destination_runtime->hostLayerState(kTargetLayer);
            destination_host.reserved_u64[2] = kNumExperts;
            destination_host.reserved_u64[3] = kPlanCapacity;
            ASSERT_TRUE(backend->hostToDeviceOnStream(
                destination_runtime->deviceLayerState(kTargetLayer),
                &destination_host,
                sizeof(destination_host),
                device.ordinal,
                stream));

            DeviceMoERebalanceConfig base_config;
            base_config.num_layers = kRuntimeLayerCount;
            base_config.num_experts = kNumExperts;
            base_config.top_k = kTopK;
            base_config.participant_count = kParticipantCount;
            base_config.window_size_tokens = 1u;
            base_config.max_hot_replicas_per_participant = kPlanCapacity;
            base_config.active_transfer_slot_capacity = kTransferSlotCount;
            base_config.transfer_slot_directory_capacity = kTransferSlotCount;
            base_config.llep_enable_balanced_skip = 0u;

            std::array<DeviceMoERebalancePlanEntry, kPlanCapacity> plans{};
            for (uint32_t plan_index = 0u;
                 plan_index < kPlanCapacity;
                 ++plan_index)
            {
                auto &plan = plans[plan_index];
                plan.op = static_cast<uint32_t>(
                    DeviceMoERebalancePlanOp::ExpertPayloadArrival);
                plan.layer = kTargetLayer;
                plan.expert = transferred_experts[plan_index];
                plan.source_participant = 0u;
                plan.destination_participant = kDestinationParticipant;
                plan.source_resident_mask = 0b01u;
                plan.flags = moe_rebalance_abi::kPlanFlagCurrentBatchLLEP;
                plan.destination_slot =
                    kPhysicalTransferSlotBase + plan_index;
                plan.payload_slot = plan_index;
            }
            auto retained_layer_plans = plans;
            for (uint32_t plan_index = 0u;
                 plan_index < kPlanCapacity;
                 ++plan_index)
            {
                retained_layer_plans[plan_index].layer = 0u;
                retained_layer_plans[plan_index].destination_slot = plan_index;
            }

            DeviceMoERebalanceCommandBufferHeader source_header;
            source_header.epoch = 1u;
            source_header.phase = static_cast<uint32_t>(
                DeviceMoERebalancePipelinePhase::PlanAssignments);
            source_header.command_count = kPlanCapacity;
            source_header.command_capacity = kPlanCapacity;
            source_header.participant_id = 0u;
            source_header.participant_count = kParticipantCount;
            auto destination_header = source_header;
            destination_header.participant_id = kDestinationParticipant;
            auto retained_source_header = source_header;
            // Both layer-local transfers belong to this same acquired batch.
            // Their disjoint physical slot ranges distinguish the arrivals;
            // inventing a second epoch would escape the live reader's lease.
            auto retained_destination_header = retained_source_header;
            retained_destination_header.participant_id =
                kDestinationParticipant;
            DeviceMoERebalanceCommandBufferHeader observed_destination_header{};

            const uint32_t plan_count = kPlanCapacity;
            auto source_config = base_config;
            source_config.participant_id = 0u;
            auto destination_config = base_config;
            destination_config.participant_id = kDestinationParticipant;
            const auto apply_config = prefillLLEPTransferConfig(
                destination_config,
                PrefillLLEPTransferPurpose::CurrentBatchMovement);

            ASSERT_TRUE(backend->memset(
                d_source_descriptors.get(),
                0,
                d_source_descriptors.bytes(),
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->memset(
                d_local_payload.get(),
                0,
                d_local_payload.bytes(),
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->memset(
                d_gathered_payload.get(),
                0,
                d_gathered_payload.bytes(),
                device.ordinal,
                stream));
            transfer_directory->resetRequestPublications(stream);

            /*
             * First retain one complete layer-0 wave in physical slots 0..31.
             * Layer 1 below must materialize into slots 32..63 without changing
             * any descriptor still published by this earlier layer.
             */
            ASSERT_TRUE(backend->hostToDeviceOnStream(
                d_plan.get(),
                retained_layer_plans.data(),
                sizeof(retained_layer_plans),
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->hostToDeviceOnStream(
                d_plan_count.get(),
                &plan_count,
                sizeof(plan_count),
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->hostToDeviceOnStream(
                d_command_header.get(),
                &retained_source_header,
                sizeof(retained_source_header),
                device.ordinal,
                stream));
            ASSERT_TRUE(kernel.packDeviceRebalanceSourceDescriptors(
                launch,
                source_runtime->deviceLayerState(0),
                d_plan.as<DeviceMoERebalancePlanEntry>(),
                d_command_header.as<DeviceMoERebalanceCommandBufferHeader>(),
                kPlanCapacity,
                d_source_descriptors.as<DeviceMoEExpertDirectoryEntry>(),
                source_config,
                /*controller_state=*/nullptr,
                kCommandBufferCount));
            ASSERT_TRUE(kernel.packDeviceRebalanceCompactPayloads(
                launch,
                d_plan.as<DeviceMoERebalancePlanEntry>(),
                d_command_header.as<DeviceMoERebalanceCommandBufferHeader>(),
                kPlanCapacity,
                d_source_descriptors.as<DeviceMoEExpertDirectoryEntry>(),
                d_local_payload.as<uint8_t>(),
                /*local_payload_slot_count=*/kPlanCapacity,
                payload_slot_bytes,
                source_config,
                d_copy_status.as<DeviceMoERebalanceApplyStatus>(),
                /*controller_state=*/nullptr,
                kCommandBufferCount));
            ASSERT_TRUE(backend->deviceCopyAsync(
                d_gathered_payload.get(),
                d_local_payload.get(),
                local_payload_bytes,
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->hostToDeviceOnStream(
                d_command_header.get(),
                &retained_destination_header,
                sizeof(retained_destination_header),
                device.ordinal,
                stream));
            ASSERT_TRUE(kernel.unpackDeviceRebalanceCollectivePayloads(
                launch,
                d_plan.as<DeviceMoERebalancePlanEntry>(),
                d_plan_count.as<uint32_t>(),
                kPlanCapacity,
                d_command_header.as<DeviceMoERebalanceCommandBufferHeader>(),
                d_gathered_payload.as<uint8_t>(),
                /*local_payload_slot_count=*/kPlanCapacity,
                payload_slot_bytes,
                transfer_directory->deviceEntries(),
                transfer_directory->slotCount(),
                destination_config,
                d_copy_status.as<DeviceMoERebalanceApplyStatus>(),
                /*controller_state=*/nullptr,
                kCommandBufferCount));
            ASSERT_TRUE(kernel.applyDeviceRebalanceArrivals(
                launch,
                destination_runtime->deviceLayerState(0),
                d_plan.as<DeviceMoERebalancePlanEntry>(),
                d_plan_count.as<uint32_t>(),
                kPlanCapacity,
                transfer_directory->deviceEntries(),
                transfer_directory->slotCount(),
                apply_config,
                d_apply_status.as<DeviceMoERebalanceApplyStatus>(),
                d_command_header.as<DeviceMoERebalanceCommandBufferHeader>(),
                /*target_layer=*/0));

            ASSERT_TRUE(backend->hostToDeviceOnStream(
                d_plan.get(),
                plans.data(),
                sizeof(plans),
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->hostToDeviceOnStream(
                d_plan_count.get(),
                &plan_count,
                sizeof(plan_count),
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->hostToDeviceOnStream(
                d_command_header.get(),
                &source_header,
                sizeof(source_header),
                device.ordinal,
                stream));
            ASSERT_TRUE(kernel.packDeviceRebalanceSourceDescriptors(
                launch,
                source_runtime->deviceLayerState(0),
                d_plan.as<DeviceMoERebalancePlanEntry>(),
                d_command_header.as<DeviceMoERebalanceCommandBufferHeader>(),
                kPlanCapacity,
                d_source_descriptors.as<DeviceMoEExpertDirectoryEntry>(),
                source_config,
                /*controller_state=*/nullptr,
                kCommandBufferCount));
            ASSERT_TRUE(kernel.packDeviceRebalanceCompactPayloads(
                launch,
                d_plan.as<DeviceMoERebalancePlanEntry>(),
                d_command_header.as<DeviceMoERebalanceCommandBufferHeader>(),
                kPlanCapacity,
                d_source_descriptors.as<DeviceMoEExpertDirectoryEntry>(),
                d_local_payload.as<uint8_t>(),
                /*local_payload_slot_count=*/kPlanCapacity,
                payload_slot_bytes,
                source_config,
                d_copy_status.as<DeviceMoERebalanceApplyStatus>(),
                /*controller_state=*/nullptr,
                kCommandBufferCount));

            /*
             * A real two-rank graph uses one NCCL/RCCL allgather here. This
             * focused one-device regression preserves its byte layout and
             * ordering while replacing only the transport primitive with D2D.
             * Source participant zero occupies gathered lane zero.
             */
            ASSERT_TRUE(backend->deviceCopyAsync(
                d_gathered_payload.get(),
                d_local_payload.get(),
                local_payload_bytes,
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->hostToDeviceOnStream(
                d_command_header.get(),
                &destination_header,
                sizeof(destination_header),
                device.ordinal,
                stream));

            ASSERT_TRUE(kernel.unpackDeviceRebalanceCollectivePayloads(
                launch,
                d_plan.as<DeviceMoERebalancePlanEntry>(),
                d_plan_count.as<uint32_t>(),
                kPlanCapacity,
                d_command_header.as<DeviceMoERebalanceCommandBufferHeader>(),
                d_gathered_payload.as<uint8_t>(),
                /*local_payload_slot_count=*/kPlanCapacity,
                payload_slot_bytes,
                transfer_directory->deviceEntries(),
                transfer_directory->slotCount(),
                destination_config,
                d_copy_status.as<DeviceMoERebalanceApplyStatus>(),
                /*controller_state=*/nullptr,
                kCommandBufferCount));

            ASSERT_TRUE(kernel.applyDeviceRebalanceArrivals(
                launch,
                destination_runtime->deviceLayerState(0),
                d_plan.as<DeviceMoERebalancePlanEntry>(),
                d_plan_count.as<uint32_t>(),
                kPlanCapacity,
                transfer_directory->deviceEntries(),
                transfer_directory->slotCount(),
                apply_config,
                d_apply_status.as<DeviceMoERebalanceApplyStatus>(),
                d_command_header.as<DeviceMoERebalanceCommandBufferHeader>(),
                /*target_layer=*/kTargetLayer));

            // Preserve the exact descriptor generation consumed by apply before
            // this one-device transport fixture reuses it for the source rank.
            // This ordered diagnostic copy adds no intermediate stream wait.
            ASSERT_TRUE(backend->deviceToHostOnStream(
                &observed_destination_header, d_command_header.get(),
                sizeof(observed_destination_header), device.ordinal, stream));

            /*
             * The source participates in the same domain-wide transaction even
             * though neither payload arrival targets participant zero. Publish
             * its independent completion record now; the split-row regression
             * below must consume the exact per-participant status object that a
             * real NCCL/RCCL graph would produce.
             */
            ASSERT_TRUE(backend->hostToDeviceOnStream(
                d_command_header.get(),
                &source_header,
                sizeof(source_header),
                device.ordinal,
                stream));
            ASSERT_TRUE(kernel.applyDeviceRebalanceArrivals(
                launch,
                source_runtime->deviceLayerState(0),
                d_plan.as<DeviceMoERebalancePlanEntry>(),
                d_plan_count.as<uint32_t>(),
                kPlanCapacity,
                transfer_directory->deviceEntries(),
                transfer_directory->slotCount(),
                prefillLLEPTransferConfig(
                    source_config,
                    PrefillLLEPTransferPurpose::CurrentBatchMovement),
                d_source_apply_status.as<DeviceMoERebalanceApplyStatus>(),
                d_command_header.as<DeviceMoERebalanceCommandBufferHeader>(),
                /*target_layer=*/kTargetLayer));

            DeviceMoERebalanceStatus transfer_status;
            transfer_status.planned_arrivals = kPlanCapacity;
            transfer_status.llep_assignment_span_count = kNumExperts;
            transfer_status.llep_weight_transfer_count = kPlanCapacity;
            ASSERT_TRUE(backend->hostToDeviceOnStream(
                d_transfer_status.get(),
                &transfer_status,
                sizeof(transfer_status),
                device.ordinal,
                stream));

            DeviceMoERebalanceApplyStatus observed_apply{};
            DeviceMoERebalanceApplyStatus observed_copy{};
            DeviceMoELayerRuntime observed_retained_runtime{};
            DeviceMoELayerRuntime observed_runtime{};
            DeviceMoELayerRuntime observed_durable_runtime{};
            DeviceMoEOverlayEpochTicket observed_reader_ticket{};
            ASSERT_TRUE(backend->deviceToHostOnStream(
                &observed_copy, d_copy_status.get(), sizeof(observed_copy),
                device.ordinal, stream));
            ASSERT_TRUE(backend->deviceToHostOnStream(
                &observed_apply,
                d_apply_status.get(),
                sizeof(observed_apply),
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->deviceToHostOnStream(
                &observed_retained_runtime,
                destination_runtime->deviceLayerState(0),
                sizeof(observed_retained_runtime),
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->deviceToHostOnStream(
                &observed_runtime,
                destination_runtime->deviceLayerState(kTargetLayer),
                sizeof(observed_runtime),
                device.ordinal,
                stream));
            ASSERT_TRUE(backend->deviceToHostOnStream(
                &observed_durable_runtime,
                destination_placement->deviceLayerState(kTargetLayer),
                sizeof(observed_durable_runtime), device.ordinal, stream));
            ASSERT_TRUE(backend->deviceToHostOnStream(
                &observed_reader_ticket, destination_epoch->requestTicket(0u),
                sizeof(observed_reader_ticket), device.ordinal, stream));
            ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));

            ASSERT_EQ(observed_apply.status_code, 0u)
                << "invalid_entries=" << observed_apply.invalid_plan_entries
                << "header magic=" << observed_destination_header.magic
                << " version=" << observed_destination_header.version
                << " phase=" << observed_destination_header.phase
                << " participant=" << observed_destination_header.participant_id
                << " participants=" << observed_destination_header.participant_count
                << " expected version=" << apply_config.version
                << " participant=" << apply_config.participant_id
                << " participants=" << apply_config.participant_count;
            ASSERT_EQ(observed_apply.invalid_plan_entries, 0u);
            ASSERT_EQ(observed_apply.missing_source_descriptors, 0u);
            ASSERT_EQ(observed_apply.missing_destination_slots, 0u);
            ASSERT_EQ(observed_apply.descriptor_mismatches, 0u);
            ASSERT_EQ(observed_apply.copy_incomplete, 0u);
            ASSERT_EQ(observed_apply.required_local_arrivals, kPlanCapacity);
            ASSERT_EQ(observed_apply.ready_local_arrivals, kPlanCapacity);
            ASSERT_EQ(observed_apply.applied_arrivals, kPlanCapacity);
            ASSERT_EQ(observed_copy.status_code, 0u);
            ASSERT_EQ(observed_copy.copied_arrivals, kPlanCapacity);
            // Compare actual destination copy bytes with the canonical exact
            // format profile, not the largest padded collective lane. This
            // catches stale accumulation and whole-capacity byte reporting for
            // every codebook on both CUDA and HIP.
            const auto exact_profile = DeviceMoETransferSlotDirectory::profileForLayerFormats(
                {format_specs[format_index]});
            EXPECT_EQ(observed_copy.copied_payload_bytes,
                      kPlanCapacity * exact_profile.max_wire_payload_bytes);

            ASSERT_LE(observed_runtime.active_bank, 1u);
            const auto &destination_bank =
                observed_runtime.banks[observed_runtime.active_bank];
            for (int expert = 0; expert < kNumExperts; ++expert)
            {
                ASSERT_EQ(destination_bank.local_compute_mask[expert], 1u)
                    << "expert=" << expert;
                ASSERT_EQ(destination_bank.resident_participant_mask[expert], 0b11u)
                    << "expert=" << expert;
                ASSERT_EQ(
                    destination_bank.experts[expert].gate.codebook_id,
                    format.device_execution_codebook_id)
                    << "expert=" << expert;
            }
            const auto &slot_entries =
                transfer_directory->hostEntriesForTest();
            ASSERT_LE(observed_retained_runtime.active_bank, 1u);
            const auto &retained_bank = observed_retained_runtime.banks[
                observed_retained_runtime.active_bank];
            for (uint32_t plan_index = 0u;
                 plan_index < kPlanCapacity;
                 ++plan_index)
            {
                const uint32_t expert = transferred_experts[plan_index];
                const auto &retained_descriptor =
                    retained_bank.experts[expert];
                const auto &retained_slot_descriptor =
                    slot_entries[plan_index].descriptor;
                const uint32_t physical_slot =
                    kPhysicalTransferSlotBase + plan_index;
                const auto &destination_descriptor =
                    destination_bank.experts[expert];
                const auto &slot_descriptor =
                    slot_entries[physical_slot].descriptor;
                ASSERT_EQ(
                    retained_descriptor.local_slot,
                    static_cast<int32_t>(plan_index));
                ASSERT_EQ(
                    retained_descriptor.gate.payload,
                    retained_slot_descriptor.gate.payload);
                ASSERT_EQ(
                    retained_descriptor.up.payload,
                    retained_slot_descriptor.up.payload);
                ASSERT_EQ(
                    retained_descriptor.down.payload,
                    retained_slot_descriptor.down.payload);
                ASSERT_EQ(
                    destination_descriptor.local_slot,
                    static_cast<int32_t>(physical_slot));
                ASSERT_EQ(
                    destination_descriptor.gate.payload,
                    slot_descriptor.gate.payload);
                ASSERT_EQ(
                    destination_descriptor.up.payload,
                    slot_descriptor.up.payload);
                ASSERT_EQ(
                    destination_descriptor.down.payload,
                    slot_descriptor.down.payload);
            }
            ASSERT_EQ(observed_runtime.current_batch_llep_movement_observed, 1u);
            ASSERT_EQ(observed_runtime.current_batch_llep_transient_bank_active, 1u);
            ASSERT_EQ(observed_reader_ticket.epoch, 1u);
            ASSERT_EQ(observed_durable_runtime.active_bank, 1u);
            ASSERT_EQ(observed_durable_runtime.active_epoch, 1u);
            ASSERT_EQ(observed_durable_runtime.current_batch_llep_transient_bank_active, 0u);
            for (const auto expert : transferred_experts)
            {
                // The copied descriptor belongs to this batch, not the durable
                // parent. Publishing it there would corrupt later requests.
                ASSERT_EQ(observed_durable_runtime.banks[1].local_compute_mask[expert], 0u);
                ASSERT_EQ(observed_durable_runtime.banks[1].resident_participant_mask[expert], 0b01u);
            }

            if (scope == NativeExpertTransferProofScope::CopyPublication)
            {
                // The model-free preflight proves real byte movement and the
                // exact reader lifetime for every codebook. The separate math
                // selection retains the complete expensive M-totality sweep.
                for (const auto &epoch : {source_epoch, destination_epoch})
                    ASSERT_TRUE(kernel.releaseMoEOverlayEpoch(
                        launch, epoch->control(), epoch->requestTicket(0u),
                        epoch->requestStatus(0u)));
                ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
                continue;
            }

            for (const int rows : kRows)
            {
                SCOPED_TRACE("M=" + std::to_string(rows));
                auto hidden = makeHidden(rows, kDModel, format_index);
                ASSERT_TRUE(hidden->ensureOnDevice(device, stream));

                auto routing_indices = TestTensorFactory::createFP32(
                    {static_cast<size_t>(rows), static_cast<size_t>(kTopK)});
                auto routing_weights = TestTensorFactory::createFP32(
                    {static_cast<size_t>(rows), static_cast<size_t>(kTopK)});
                constexpr std::array<float, kTopK> kRouteWeights = {
                    0.25f, 0.20f, 0.16f, 0.13f,
                    0.10f, 0.07f, 0.05f, 0.04f};
                for (int row = 0; row < rows; ++row)
                {
                    for (int route = 0; route < kTopK; ++route)
                    {
                        const size_t slot =
                            static_cast<size_t>(row) * kTopK + route;
                        routing_indices->mutable_data()[slot] =
                            static_cast<float>((row + route) % kNumExperts);
                        routing_weights->mutable_data()[slot] =
                            kRouteWeights[route];
                    }
                }
                ASSERT_TRUE(routing_indices->ensureOnDevice(device, stream));
                ASSERT_TRUE(routing_weights->ensureOnDevice(device, stream));

                ASSERT_TRUE(kernel.publishCompleteGroupedPrefillPlanFromRouter(
                    source_runtime->deviceLayerState(kTargetLayer),
                    routing_indices.get(),
                    routing_weights.get(),
                    rows,
                    rows,
                    kNumExperts,
                    kTopK,
                    source_gateup_table,
                    source_down_table,
                    /*filter_to_local_runtime_experts=*/true));
                std::vector<float> static_owner_canonical;
                const auto static_owner_output = executePublishedPlan(
                    backend,
                    kernel,
                    *source_runtime,
                    device,
                    stream,
                    hidden.get(),
                    source_gateup_table,
                    source_down_table,
                    rows,
                    kDModel,
                    kIntermediate,
                    kNumExperts,
                    kTopK,
                    kTargetLayer,
                    &static_owner_canonical);

                ASSERT_TRUE(kernel.groupPrefillRoutes(
                    destination_runtime->deviceLayerState(kTargetLayer),
                    routing_indices.get(),
                    routing_weights.get(),
                    rows,
                    rows,
                    kNumExperts,
                    kTopK,
                    /*filter_to_local_runtime_experts=*/false));

                std::array<least_loaded_ep::LeastLoadedExpertAssignmentSpan,
                           kNumExperts>
                    spans{};
                std::array<int32_t, kNumExperts * 2> span_bounds{};
                for (int expert = 0; expert < kNumExperts; ++expert)
                {
                    spans[expert] = {
                        .expert = static_cast<uint32_t>(expert),
                        .owner_participant = 0u,
                        .destination_participant = kDestinationParticipant,
                        .route_row_begin = 0u,
                        .route_row_end = static_cast<uint64_t>(rows),
                        .needs_foreign_weight = static_cast<uint8_t>(
                            (expert & 1) != 0 ? 1u : 0u),
                    };
                    span_bounds[2 * expert] = expert;
                    span_bounds[2 * expert + 1] = expert + 1;
                }
                ASSERT_TRUE(backend->hostToDeviceOnStream(
                    destination_host.reserved_ptrs[1],
                    spans.data(),
                    sizeof(spans),
                    device.ordinal,
                    stream));
                ASSERT_TRUE(backend->hostToDeviceOnStream(
                    destination_host.reserved_ptrs[0],
                    span_bounds.data(),
                    sizeof(span_bounds),
                    device.ordinal,
                    stream));
                ASSERT_TRUE(
                    kernel.assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers(
                        launch,
                        destination_runtime->deviceLayerState(kTargetLayer),
                        rows,
                        rows,
                        kNumExperts,
                        kTopK,
                        d_transfer_status.as<DeviceMoERebalanceStatus>(),
                        d_apply_status.as<DeviceMoERebalanceApplyStatus>()));

                std::vector<int32_t> assigned_participants(
                    static_cast<size_t>(rows) * kTopK,
                    -1);
                ASSERT_TRUE(backend->deviceToHostOnStream(
                    assigned_participants.data(),
                    destination_host.route_participant_ids,
                    assigned_participants.size() * sizeof(int32_t),
                    device.ordinal,
                    stream));
                ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
                ASSERT_TRUE(std::all_of(
                    assigned_participants.begin(),
                    assigned_participants.end(),
                    [=](int32_t participant)
                    {
                        return participant ==
                               static_cast<int32_t>(kDestinationParticipant);
                    })) << "CurrentBatchLLEP did not assign every routed slot to the ready destination";

                ASSERT_TRUE(
                    kernel.publishCompleteGroupedPrefillPlanFromRuntimeAssignments(
                        destination_runtime->deviceLayerState(kTargetLayer),
                        rows,
                        rows,
                        kNumExperts,
                        kTopK,
                        destination_gateup_table,
                        destination_down_table));
                std::vector<float> llep_canonical;
                const auto llep_output = executePublishedPlan(
                    backend,
                    kernel,
                    *destination_runtime,
                    device,
                    stream,
                    hidden.get(),
                    destination_gateup_table,
                    destination_down_table,
                    rows,
                    kDModel,
                    kIntermediate,
                    kNumExperts,
                    kTopK,
                    kTargetLayer,
                    &llep_canonical);

                double norm_squared = 0.0;
                for (const float value : static_owner_output)
                {
                    ASSERT_TRUE(std::isfinite(value));
                    norm_squared += static_cast<double>(value) * value;
                }
                ASSERT_GT(norm_squared, 1.0e-14)
                    << format.label << " produced a degenerate all-zero witness";
                expectByteEqual(
                    std::string(backend_label) + " " + format.label +
                        " transferred CurrentBatchLLEP vs StaticOwner M=" +
                        std::to_string(rows),
                    llep_output,
                    static_owner_output,
                    kDModel);
                expectByteEqual(
                    std::string(backend_label) + " " + format.label +
                        " transferred CurrentBatchLLEP canonical routes vs StaticOwner M=" +
                        std::to_string(rows),
                    llep_canonical,
                    static_owner_canonical,
                    kDModel);

                /*
                 * Exercise the defining LLEP shape that the complete-expert
                 * comparison above cannot cover: every logical expert is split
                 * into two contiguous grouped-row spans and both participants
                 * execute one half. Production publishes those partials into
                 * the same canonical [row, route, column] coordinates before a
                 * rooted collective. Comparing the selected route slot before
                 * reduction isolates assignment/regrouping from collective
                 * arithmetic and proves that no row is duplicated, omitted, or
                 * associated with the wrong route weight.
                 */
                constexpr size_t kSplitSpanCount =
                    static_cast<size_t>(kNumExperts) * 2u;
                std::array<least_loaded_ep::LeastLoadedExpertAssignmentSpan,
                           kSplitSpanCount>
                    split_spans{};
                std::array<int32_t, kNumExperts * 2> split_span_bounds{};
                std::array<uint32_t, kNumExperts> grouped_route_counts{};
                for (int row = 0; row < rows; ++row)
                {
                    for (int route = 0; route < kTopK; ++route)
                    {
                        const int expert = (row + route) % kNumExperts;
                        ++grouped_route_counts[expert];
                    }
                }
                for (int expert = 0; expert < kNumExperts; ++expert)
                {
                    const size_t first = static_cast<size_t>(expert) * 2u;
                    const uint32_t route_count =
                        grouped_route_counts[expert];
                    const uint32_t split_row = (route_count + 1u) / 2u;
                    split_spans[first] = {
                        .expert = static_cast<uint32_t>(expert),
                        .owner_participant = 0u,
                        .destination_participant = 0u,
                        .route_row_begin = 0u,
                        .route_row_end = split_row,
                        .needs_foreign_weight = 0u,
                    };
                    split_spans[first + 1u] = {
                        .expert = static_cast<uint32_t>(expert),
                        .owner_participant = 0u,
                        .destination_participant = kDestinationParticipant,
                        .route_row_begin = split_row,
                        .route_row_end = route_count,
                        .needs_foreign_weight = static_cast<uint8_t>(
                            (expert & 1) != 0 ? 1u : 0u),
                    };
                    split_span_bounds[2 * expert] =
                        static_cast<int32_t>(first);
                    split_span_bounds[2 * expert + 1] =
                        static_cast<int32_t>(first + 2u);
                }

                for (auto *runtime : {source_runtime.get(),
                                      destination_runtime.get()})
                {
                    ASSERT_TRUE(kernel.groupPrefillRoutes(
                        runtime->deviceLayerState(kTargetLayer),
                        routing_indices.get(),
                        routing_weights.get(),
                        rows,
                        rows,
                        kNumExperts,
                        kTopK,
                        /*filter_to_local_runtime_experts=*/false));
                    ASSERT_TRUE(publishCurrentBatchPlanCounts(
                        backend,
                        device,
                        stream,
                        runtime->deviceLayerState(kTargetLayer),
                        kSplitSpanCount,
                        kPlanCapacity));
                    auto &host_runtime = runtime->hostLayerState(kTargetLayer);
                    ASSERT_TRUE(backend->hostToDeviceOnStream(
                        host_runtime.reserved_ptrs[1],
                        split_spans.data(),
                        sizeof(split_spans),
                        device.ordinal,
                        stream));
                    ASSERT_TRUE(backend->hostToDeviceOnStream(
                        host_runtime.reserved_ptrs[0],
                        split_span_bounds.data(),
                        sizeof(split_span_bounds),
                        device.ordinal,
                        stream));
                }

                transfer_status.llep_assignment_span_count =
                    static_cast<uint32_t>(kSplitSpanCount);
                ASSERT_TRUE(backend->hostToDeviceOnStream(
                    d_transfer_status.get(),
                    &transfer_status,
                    sizeof(transfer_status),
                    device.ordinal,
                    stream));
                ASSERT_TRUE(
                    kernel.assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers(
                        launch,
                        source_runtime->deviceLayerState(kTargetLayer),
                        rows,
                        rows,
                        kNumExperts,
                        kTopK,
                        d_transfer_status.as<DeviceMoERebalanceStatus>(),
                        d_source_apply_status.as<DeviceMoERebalanceApplyStatus>()));
                ASSERT_TRUE(
                    kernel.assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers(
                        launch,
                        destination_runtime->deviceLayerState(kTargetLayer),
                        rows,
                        rows,
                        kNumExperts,
                        kTopK,
                        d_transfer_status.as<DeviceMoERebalanceStatus>(),
                        d_apply_status.as<DeviceMoERebalanceApplyStatus>()));

                std::vector<int32_t> source_assignments(
                    static_cast<size_t>(rows) * kTopK,
                    -1);
                std::vector<int32_t> destination_assignments(
                    source_assignments.size(),
                    -1);
                ASSERT_TRUE(backend->deviceToHostOnStream(
                    source_assignments.data(),
                    source_host.route_participant_ids,
                    source_assignments.size() * sizeof(int32_t),
                    device.ordinal,
                    stream));
                ASSERT_TRUE(backend->deviceToHostOnStream(
                    destination_assignments.data(),
                    destination_host.route_participant_ids,
                    destination_assignments.size() * sizeof(int32_t),
                    device.ordinal,
                    stream));
                ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
                ASSERT_EQ(source_assignments, destination_assignments)
                    << "LLEP participants consumed different assignment spans";
                ASSERT_TRUE(std::all_of(
                    source_assignments.begin(),
                    source_assignments.end(),
                    [](int32_t participant)
                    {
                        return participant == 0 || participant == 1;
                    }));
                ASSERT_NE(
                    std::find(source_assignments.begin(),
                              source_assignments.end(),
                              0),
                    source_assignments.end());
                ASSERT_NE(
                    std::find(source_assignments.begin(),
                              source_assignments.end(),
                              1),
                    source_assignments.end());

                ASSERT_TRUE(
                    kernel.publishCompleteGroupedPrefillPlanFromRuntimeAssignments(
                        source_runtime->deviceLayerState(kTargetLayer),
                        rows,
                        rows,
                        kNumExperts,
                        kTopK,
                        source_gateup_table,
                        source_down_table));

                /*
                 * A production participant owns a distinct kernel instance and
                 * grouping workspace. This focused fixture intentionally reuses
                 * one physical GPU and one kernel instance for both simulated
                 * participants, so each participant's publication and execution
                 * must remain one indivisible test transaction. Publishing the
                 * destination first would overwrite the source participant's
                 * inverse map in the shared fixture workspace and manufacture a
                 * cross-participant race that cannot exist between real ranks.
                 */
                std::vector<float> source_split_canonical;
                const auto source_split_output = executePublishedPlan(
                    backend,
                    kernel,
                    *source_runtime,
                    device,
                    stream,
                    hidden.get(),
                    source_gateup_table,
                    source_down_table,
                    rows,
                    kDModel,
                    kIntermediate,
                    kNumExperts,
                    kTopK,
                    kTargetLayer,
                    &source_split_canonical);

                ASSERT_TRUE(
                    kernel.publishCompleteGroupedPrefillPlanFromRuntimeAssignments(
                        destination_runtime->deviceLayerState(kTargetLayer),
                        rows,
                        rows,
                        kNumExperts,
                        kTopK,
                        destination_gateup_table,
                        destination_down_table));

                std::vector<float> destination_split_canonical;
                const auto destination_split_output = executePublishedPlan(
                    backend,
                    kernel,
                    *destination_runtime,
                    device,
                    stream,
                    hidden.get(),
                    destination_gateup_table,
                    destination_down_table,
                    rows,
                    kDModel,
                    kIntermediate,
                    kNumExperts,
                    kTopK,
                    kTargetLayer,
                    &destination_split_canonical);
                (void)source_split_output;
                (void)destination_split_output;

                ASSERT_EQ(source_split_canonical.size(),
                          static_owner_canonical.size());
                ASSERT_EQ(destination_split_canonical.size(),
                          static_owner_canonical.size());
                std::vector<float> reconstructed_canonical(
                    static_owner_canonical.size(),
                    0.0f);
                for (size_t route_slot = 0;
                     route_slot < source_assignments.size();
                     ++route_slot)
                {
                    const bool source_selected =
                        source_assignments[route_slot] == 0;
                    const auto &selected =
                        source_selected
                            ? source_split_canonical
                            : destination_split_canonical;
                    const auto &inactive =
                        source_selected
                            ? destination_split_canonical
                            : source_split_canonical;
                    const size_t begin =
                        route_slot * static_cast<size_t>(kDModel);
                    const size_t end = begin + static_cast<size_t>(kDModel);
                    std::copy(
                        selected.begin() + static_cast<std::ptrdiff_t>(begin),
                        selected.begin() + static_cast<std::ptrdiff_t>(end),
                        reconstructed_canonical.begin() +
                            static_cast<std::ptrdiff_t>(begin));
                    for (size_t element = begin; element < end; ++element)
                    {
                        uint32_t inactive_bits = 0u;
                        std::memcpy(
                            &inactive_bits,
                            &inactive[element],
                            sizeof(inactive_bits));
                        ASSERT_EQ(inactive_bits & 0x7fffffffu, 0u)
                            << "inactive participant wrote canonical route slot="
                            << route_slot << " element=" << element;
                    }
                }
                expectByteEqual(
                    std::string(backend_label) + " " + format.label +
                        " participant-split CurrentBatchLLEP canonical routes vs StaticOwner M=" +
                        std::to_string(rows),
                    reconstructed_canonical,
                    static_owner_canonical,
                    kDModel);
            }
            for (const auto &epoch : {source_epoch, destination_epoch})
                ASSERT_TRUE(kernel.releaseMoEOverlayEpoch(
                    launch, epoch->control(), epoch->requestTicket(0u),
                    epoch->requestStatus(0u)));
            // Descriptors, transfer slots and both arenas outlive the reader's
            // release. This terminal fixture wait is never a production edge.
            ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
        }

        ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
        workspace_consumer->unbindWorkspace();
    }
}
