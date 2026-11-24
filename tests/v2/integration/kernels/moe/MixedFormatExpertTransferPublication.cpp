/**
 * @file MixedFormatExpertTransferPublication.cpp
 * @brief Captured CUDA/ROCm movement across exact expert slot families.
 *
 * The production pack/unpack kernels run in one retained graph. Only setup
 * inputs change between replays; each family keeps its own physical address
 * and immutable layout while its occupants change. Stale lease replays must
 * leave the previous publication untouched. Arithmetic parity is certified by
 * the separate grouped-expert sweeps; this test isolates the transfer contract.
 * Original IQ3_S/IQ4_XS coefficient planes have independently fixed byte
 * extents so a production metadata omission cannot also shrink the oracle.
 */
#include "NativeQuantisedExpertTransferParityTest.h"
#include "execution/config/RuntimeConfig.h"
#include <limits>
#ifdef HAVE_ROCM
#include "backends/rocm/ROCmHardwareMetadata.h"
#endif

namespace llaminar2::test
{
    namespace
    {
        /** @brief Replay one live layer whose family has durable claims outside the wave. */
        void testTransferFamilyOccupancy(DeviceId device, void *stream, IBackend *backend, IMoEKernel &kernel)
        {
            using native_quantised_transfer_parity_detail::DeviceAllocation;
            namespace abi = moe_rebalance_abi;
            constexpr uint32_t layers = 3u, experts = 300u, live_layers = 1u;
            constexpr size_t words = abi::histogramWords(abi::HistogramLayout::TransferFamilyOccupancy, live_layers, experts);
            constexpr std::array ranges{
                DeviceMoETransferSlotRange{0u, 601u, 600u},
                DeviceMoETransferSlotRange{601u, 903u, 300u},
                DeviceMoETransferSlotRange{0u, 601u, 600u}};
            DeviceAllocation device_ranges(backend, device.ordinal, sizeof(ranges));
            DeviceAllocation device_runtime(backend, device.ordinal, layers * sizeof(DeviceMoELayerRuntime));
            DeviceAllocation device_words(backend, device.ordinal, (words + 1u) * sizeof(uint32_t));
            const auto upload = [&](void *dst, const void *src, size_t bytes) {
                ASSERT_TRUE(backend->hostToDeviceOnStream(dst, src, bytes, device.ordinal, stream));
            };
            ASSERT_NO_FATAL_FAILURE(upload(device_ranges.get(), ranges.data(), sizeof(ranges)));
            std::vector<DeviceMoELayerRuntime> runtime(layers);
            for (uint32_t layer = 0; layer < layers; ++layer)
            {
                auto &local = runtime[layer];
                local.active_epoch = 1u;
                local.expert_count = experts;
                local.top_k = 1u;
                local.participant_count = 2u;
                if (layer == 1u)
                    continue; // An empty layer in the other family still emits exactly one metadata word when scheduled.
                for (uint32_t expert = 0u; expert < experts; ++expert)
                {
                    auto &desc = local.banks[0].experts[expert];
                    desc.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid | DeviceMoEExpertFlags::Resident |
                        DeviceMoEExpertFlags::TransferSlot);
                    desc.owner_participant = 0;
                    desc.local_slot = static_cast<int32_t>((layer == 0 ? 0u : experts) + expert);
                    local.banks[0].resident_participant_mask[expert] = 1u;
                }
            }
            std::array<uint32_t, words + 1u> evidence{};
            evidence.back() = 0x1a2b3c4du;
            ASSERT_NO_FATAL_FAILURE(upload(device_words.get(), evidence.data(), sizeof(evidence)));
            DeviceMoERebalanceConfig config;
            config.num_layers = layers;
            config.num_experts = experts;
            config.top_k = 1u;
            config.participant_count = 2u;
            config.window_size_tokens = 1u;
            config.layer_wave_count = live_layers;
            config.active_transfer_slot_capacity = 900u;
            config.transfer_slot_directory_capacity = 903u;
            config.transfer_slot_layer_ranges = device_ranges.as<DeviceMoETransferSlotRange>();
            config.histogram_layout = moe_rebalance_abi::HistogramLayout::TransferFamilyOccupancy;
            auto &context = GPUDeviceContextPool::instance().getContext(device);
            auto graph = context.createGraphCapture(stream);
            ASSERT_NE(graph, nullptr);
            ScopedBackendGraphCapture capture(context, *graph, "partial-wave exact family occupancy");
            ASSERT_TRUE(capture.begin());
            const bool packed = kernel.packDeviceRebalanceHistograms({.stream = stream},
                device_runtime.as<DeviceMoELayerRuntime>(), device_words.as<uint64_t>(), config);
            capture.finish();
            ASSERT_TRUE(packed);
            ASSERT_TRUE(graph->instantiate());
            for (const uint32_t expected : {600u, 0u})
            {
                if (expected == 0u)
                    for (auto &layer : runtime)
                        for (uint32_t expert = 0; expert < experts; ++expert)
                        {
                            layer.banks[0].resident_participant_mask[expert] = 0u;
                            layer.banks[0].experts[expert].flags = 0u;
                        }
                ASSERT_NO_FATAL_FAILURE(upload(device_runtime.get(), runtime.data(), runtime.size() * sizeof(runtime[0])));
                ASSERT_TRUE(graph->launch());
                ASSERT_TRUE(backend->deviceToHostOnStream(evidence.data(), device_words.get(), sizeof(evidence), device.ordinal, stream));
                ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
                EXPECT_EQ(evidence[2u * experts], expected);
                EXPECT_EQ(evidence.back(), 0x1a2b3c4du) << "Partial wave must not emit occupancy for unscheduled layers";
            }
        }
    }

    void runMixedFormatExpertTransferPublication(DeviceId device, void *stream,
        DeviceMoEProjectionSet projections)
    {
        using namespace native_quantised_transfer_parity_detail;
        ASSERT_TRUE(device.is_gpu());
        ASSERT_NE(stream, nullptr);
        auto *backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        enum class Packing { Compact, Collective };
        for (const auto packing : {Packing::Compact, Packing::Collective})
        {
            SCOPED_TRACE(packing == Packing::Compact ? "compact" : "collective");
            constexpr int n = 64, k = 64;
            constexpr size_t plane_bytes = n * k * sizeof(float);
            constexpr size_t source_bytes = 3u * 4u * plane_bytes;
            using Specs = std::vector<DeviceMoETransferSlotDirectory::ProjectionSpec>;
            std::vector<Specs> formats;
            for (const auto &format : quantizedMoEVerifierFormats())
                formats.push_back(projectionSpecsForFormat(format, k, n));
            // Include distinct floating precisions and reuse one coded/quantized family.
            for (const auto type : {TensorType::FP32, TensorType::FP16, TensorType::BF16,
                                    TensorType::FP32})
            {
                Specs specs;
                for (const char *label : {"gate", "up", "down"})
                    specs.push_back({.label = label, .N = n, .K = k,
                                     .format = ExpertWeightFormat::floating(type)});
                formats.push_back(std::move(specs));
            }
            formats.push_back(formats.front());
            // Identical all-format replay inventory for each ownership model.
            // Pair directories physically omit down storage rather than merely
            // declining to transmit an allocated whole-expert buffer.
            if (projections == DeviceMoEProjectionSet::GateUp)
                for (auto &specs : formats) specs.pop_back();
            const auto profile = DeviceMoETransferSlotDirectory::profileForLayerFormats(formats, projections);
            DeviceMoERebalanceConfig capacity_policy;
            capacity_policy.num_layers = static_cast<uint32_t>(formats.size());
            const PrefixCacheRuntimeConfig prefix_cache;
            const auto minimum = DeviceMoETransferSlotDirectory::runtimeMinimumActiveSlots(true,
                RoutedExpertAssignmentPolicy::StaticOwner, prefix_cache, 3u);
            ASSERT_EQ(minimum, 3u);
            const auto capacities = DeviceMoETransferSlotDirectory::planRuntimeCapacity(capacity_policy, minimum, 1u, 1u);
            const auto quote = DeviceMoETransferSlotDirectory::allocationBOM(capacities, profile);
            const PhysicalMemoryResource resource{0, device, quote.total_bytes, quote.total_bytes};
            PhysicalMemoryPlanBuilder admitted;
            size_t family_bytes = 0u;
            for (const auto &family : quote.families)
            {
                admitted.add(resource, PhysicalMemoryOwner::ExpertMigrationStaging, family.totalBytes());
                family_bytes += family.totalBytes();
            }
            admitted.add(resource, PhysicalMemoryOwner::ExpertMigrationStaging, quote.layer_range_bytes);
            const auto authority = std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(admitted.build()), 0);
            auto directory = DeviceMoETransferSlotDirectory::create(
                backend, device, device.ordinal, 1u, capacities, profile, authority);
            ASSERT_NE(directory, nullptr);
            EXPECT_EQ(quote.total_bytes, family_bytes + quote.layer_range_bytes);
            EXPECT_EQ(directory->materializedBOM().total_bytes, quote.total_bytes);
            EXPECT_EQ(authority->claimedBytes(device, PhysicalMemoryOwner::ExpertMigrationStaging,
                PhysicalMemoryMaterializationKind::NewAllocation), quote.total_bytes);
            const auto prefill_capacity = DeviceMoETransferSlotDirectory::planRuntimeCapacity(capacity_policy,
                DeviceMoETransferSlotDirectory::runtimeMinimumActiveSlots(true,
                    RoutedExpertAssignmentPolicy::LeastLoadedResident, prefix_cache, 3u), 1u, 1u);
            EXPECT_NO_THROW(directory->requireCaptureIdentity(profile, prefill_capacity));
            EXPECT_THROW(directory->requireCaptureIdentity(profile,
                DeviceMoETransferSlotDirectory::planRuntimeCapacity(capacity_policy, minimum + 1u, 1u, 1u)),
                std::logic_error);
            auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
            ASSERT_NE(kernel, nullptr);
            kernel->setGPUStream(stream);
            if (packing == Packing::Compact && projections == DeviceMoEProjectionSet::CompleteExpert)
                ASSERT_NO_FATAL_FAILURE(testTransferFamilyOccupancy(device, stream, backend, *kernel));
            const MoEKernelLaunchContext launch{.stream = stream};
            DeviceMoERebalanceConfig source_config;
            source_config.num_layers = static_cast<uint32_t>(formats.size());
            source_config.num_experts = 1u;
            source_config.top_k = 1u;
            source_config.participant_count = 2u;
            source_config.participant_id = 0u;
            source_config.window_size_tokens = 1u;
            source_config.active_transfer_slot_capacity = directory->activeSlotCount();
            source_config.transfer_slot_directory_capacity = directory->slotCount();
            auto destination_config = source_config;
            destination_config.participant_id = 1u;
            destination_config.transfer_slot_layer_ranges = directory->deviceLayerRanges();
            destination_config.histogram_layout = moe_rebalance_abi::HistogramLayout::TransferFamilyOccupancy;
            const size_t payload_bytes = sizeof(DeviceMoEExpertDirectoryEntry) + directory->wirePayloadBytes();
            DeviceAllocation source_weights(backend, device.ordinal, source_bytes);
            DeviceAllocation source_runtime(backend, device.ordinal, formats.size() * sizeof(DeviceMoELayerRuntime));
            DeviceAllocation source_directory(backend, device.ordinal, 2u * formats.size() * sizeof(DeviceMoEExpertDirectoryEntry));
            DeviceAllocation plans(backend, device.ordinal, 2u * sizeof(DeviceMoERebalancePlanEntry));
            DeviceAllocation header(backend, device.ordinal, 2u * sizeof(DeviceMoERebalanceCommandBufferHeader));
            DeviceAllocation destination_header(backend, device.ordinal, sizeof(DeviceMoERebalanceCommandBufferHeader));
            DeviceAllocation payload(backend, device.ordinal, 2u * payload_bytes);
            DeviceAllocation status(backend, device.ordinal, sizeof(DeviceMoERebalanceApplyStatus));
            DeviceAllocation arrival_scratch(backend, device.ordinal, deviceMoETransferSlotClaimIndexBytes(directory->slotCount()));
            auto upload = [&](void *dst, const void *src, size_t bytes)
            {
                ASSERT_TRUE(backend->hostToDeviceOnStream(dst, src, bytes, device.ordinal, stream));
            };
            auto download = [&](void *dst, const void *src, size_t bytes)
            {
                ASSERT_TRUE(backend->deviceToHostOnStream(dst, src, bytes, device.ordinal, stream));
                ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
            };
            const std::vector<uint8_t> empty_runtime(formats.size() * sizeof(DeviceMoELayerRuntime));
            ASSERT_NO_FATAL_FAILURE(upload(source_runtime.get(), empty_runtime.data(), empty_runtime.size()));
            std::vector<uint8_t> pattern(source_bytes);
            for (size_t i = 0; i < pattern.size(); ++i)
                pattern[i] = static_cast<uint8_t>((i * 53u + i / 257u) & 255u);
            ASSERT_NO_FATAL_FAILURE(upload(source_weights.get(), pattern.data(), pattern.size()));
            DeviceMoEExpertDirectoryEntry observed;
            auto &context = GPUDeviceContextPool::instance().getContext(device);
            auto graph = context.createGraphCapture(stream);
            ASSERT_NE(graph, nullptr);
            ScopedBackendGraphCapture capture(context, *graph, "all-format expert transfer");
            ASSERT_TRUE(capture.begin());
            // Read the source from the real graph-facing placement bank. This
            // proves the GPU descriptor exporter, not just a host-built packet,
            // understands the movable projection contract.
            const bool described = packing == Packing::Compact
                ? kernel->packDeviceRebalanceSourceDescriptors(launch,
                    source_runtime.as<DeviceMoELayerRuntime>(),
                    plans.as<DeviceMoERebalancePlanEntry>(),
                    header.as<DeviceMoERebalanceCommandBufferHeader>(), 1u,
                    source_directory.as<DeviceMoEExpertDirectoryEntry>(), source_config)
                : kernel->packDeviceRebalanceDirectory(launch,
                    source_runtime.as<DeviceMoELayerRuntime>(),
                    source_directory.as<DeviceMoEExpertDirectoryEntry>(), source_config);
            const bool packed = packing == Packing::Compact ? kernel->packDeviceRebalanceCompactPayloads(
                launch, plans.as<DeviceMoERebalancePlanEntry>(),
                header.as<DeviceMoERebalanceCommandBufferHeader>(), 1u,
                source_directory.as<DeviceMoEExpertDirectoryEntry>(), payload.as<uint8_t>(),
                2u, payload_bytes, source_config, status.as<DeviceMoERebalanceApplyStatus>(), nullptr, 1u)
                : kernel->packDeviceRebalanceCollectivePayloads(
                    launch, plans.as<DeviceMoERebalancePlanEntry>(),
                    header.as<DeviceMoERebalanceCommandBufferHeader>(), 1u,
                    source_directory.as<DeviceMoEExpertDirectoryEntry>(), payload.as<uint8_t>(),
                    2u, payload_bytes, source_config, status.as<DeviceMoERebalanceApplyStatus>(), nullptr, 1u);
            const bool unpacked = kernel->unpackDeviceRebalanceCollectivePayloads(
                launch, plans.as<DeviceMoERebalancePlanEntry>(), nullptr, 1u,
                destination_header.as<DeviceMoERebalanceCommandBufferHeader>(),
                payload.as<uint8_t>(), 2u, payload_bytes, directory->deviceEntries(),
                directory->slotCount(), destination_config, status.as<DeviceMoERebalanceApplyStatus>(),
                arrival_scratch.as<DeviceMoERebalanceArrivalScratch>(), 1u);
            capture.finish();
            ASSERT_TRUE(described);
            ASSERT_TRUE(packed);
            ASSERT_TRUE(unpacked);
#ifdef HAVE_ROCM
            if (device.is_rocm())
            {
                std::vector<GPUGraphKernelNodeInfo> nodes;
                std::string error;
                ASSERT_TRUE(graph->inspectKernelNodes(nodes, &error)) << error;
                const auto unpack_node = std::find_if(nodes.begin(), nodes.end(), [](const auto &node) {
                    return node.name.find("unpack_rebalance_collective_payloads_kernel") != std::string::npos;
                });
                ASSERT_NE(unpack_node, nodes.end());
                const auto cus = rocmHardwareMetadata().at(device.ordinal).compute_units;
                EXPECT_GE(cus * unpack_node->max_active_blocks_per_sm, 1u);
                EXPECT_EQ(unpack_node->grid_x, 1u) << "The captured plan admits only one distinct arrival";
                EXPECT_EQ(unpack_node->grid_y, 1u);
                EXPECT_EQ(unpack_node->grid_z, 1u);
                EXPECT_EQ(std::count_if(nodes.begin(), nodes.end(), [](const auto &node) {
                    return node.name.find("unpack_rebalance_collective_payloads_kernel") != std::string::npos;
                }), 1);
            }
#endif
            ASSERT_TRUE(graph->instantiate());

            const auto replay_empty_arrival_wave = [&] {
                DeviceMoERebalanceApplyStatus reset{};
                DeviceMoERebalanceCommandBufferHeader empty;
                empty.participant_count = 2u;
                empty.command_capacity = 1u;
                const std::array source_commands{empty, empty};
                empty.participant_id = 1u;
                ASSERT_NO_FATAL_FAILURE(upload(status.get(), &reset, sizeof(reset)));
                ASSERT_NO_FATAL_FAILURE(upload(header.get(), source_commands.data(), sizeof(source_commands)));
                ASSERT_NO_FATAL_FAILURE(upload(destination_header.get(), &empty, sizeof(empty)));
                ASSERT_TRUE(graph->launch());
                std::uint32_t live = std::numeric_limits<std::uint32_t>::max();
                DeviceMoERebalanceApplyStatus actual;
                ASSERT_NO_FATAL_FAILURE(download(&live, arrival_scratch.get(), sizeof(live)));
                ASSERT_NO_FATAL_FAILURE(download(&actual, status.get(), sizeof(actual)));
                EXPECT_EQ(live, 0u);
                EXPECT_EQ(actual.copied_arrivals, 0u);
                EXPECT_EQ(actual.copied_payload_bytes, 0u);
            };
            ASSERT_NO_FATAL_FAILURE(replay_empty_arrival_wave());
            uint32_t epoch = 0;
            for (const auto &specs : formats)
            {
                const uint32_t layer = epoch++;
                SCOPED_TRACE(device.to_string() + " transfer epoch=" + std::to_string(epoch));
                const auto slot = directory->materializedBOM().families[profile.layer_families[layer]].range.first;
                const auto *destination_entry = directory->deviceEntries() + slot;
                ASSERT_NO_FATAL_FAILURE(download(&observed, destination_entry, sizeof(observed)));
                const auto original = observed.descriptor;
                DeviceMoEExpertDirectoryEntry source;
                source.participant = 0u;
                source.layer = layer;
                source.resident_mask = 1u;
                source.slot_index = 0u;
                source.flags = static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Valid) |
                               static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Resident);
                source.descriptor.logical_expert_id = 0;
                source.descriptor.projection_set = projections;
                source.descriptor.owner_participant = 0;
                source.descriptor.local_slot = 0;
                source.descriptor.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                                           DeviceMoEExpertFlags::Resident);
                const std::array<DeviceNativeQuantisedMatrixDesc *, 3> matrices{
                    &source.descriptor.gate, &source.descriptor.up, &source.descriptor.down};
                const std::array<DeviceMoEFloatingMatrixDesc *, 3> raw{
                    &source.descriptor.floating_gate, &source.descriptor.floating_up,
                    &source.descriptor.floating_down};
                for (size_t projection = 0; projection < specs.size(); ++projection)
                {
                    const auto &spec = specs[projection];
                    auto *base = source_weights.as<uint8_t>() + projection * 4u * plane_bytes;
                    if (spec.format.isFloating())
                    {
                        source.descriptor.weight_format = spec.format.kind == ExpertWeightFormatKind::FP32
                            ? DeviceMoEWeightFormat::FP32 : spec.format.kind == ExpertWeightFormatKind::FP16
                            ? DeviceMoEWeightFormat::FP16 : DeviceMoEWeightFormat::BF16;
                        *raw[projection] = {base, n, k};
                        continue;
                    }
                    // Source tuples own two distinct words per block even though
                    // neither format performs an additive minimum correction.
                    // Fix these extents independently of the production metadata.
                    if (!spec.codedDeclaration() && (spec.codebook_id == 11u || spec.codebook_id == kNativeQuantisedIQ4XSSourceScaleCodebook))
                    {
                        EXPECT_EQ(spec.payload_bytes_per_block, spec.codebook_id == 11u ? 13u : 16u);
                        ASSERT_TRUE(spec.is_asymmetric) << "Original source coefficient plane is mandatory";
                        EXPECT_FALSE(spec.has_emins);
                    }
                    auto &matrix = *matrices[projection];
                    matrix.payload = base;
                    matrix.scales = base + plane_bytes;
                    matrix.mins = spec.is_asymmetric ? base + 2u * plane_bytes : nullptr;
                    matrix.emins = spec.has_emins ? base + 3u * plane_bytes : nullptr;
                    matrix.n = n;
                    matrix.k = k;
                    matrix.blocks_per_row = k / 32;
                    matrix.codebook_id = spec.codebook_id;
                    matrix.source_codebook_id = spec.format.native_quantised.codebook_id;
                    matrix.source_is_superblock = spec.format.native_quantised.is_superblock;
                    matrix.source_identity_present = 1u;
                    matrix.allocation_payload_bytes_per_block = spec.payload_bytes_per_block;
                    matrix.allocation_has_mins = spec.is_asymmetric;
                    matrix.allocation_has_emins = spec.has_emins;
                    if (const auto coded = spec.codedDeclaration())
                    {
                        matrix.metadata_layout = nativeQuantisedMetadataLayout(
                            NativeQuantisedMetadataRepresentation::SuperblockCoded, coded->superblock_phase);
                        if (native_quantised_superblock::superblockCodeInPayload(spec.codebook_id))
                            matrix.scales = matrix.payload;
                        matrix.mins = base + 2u * plane_bytes;
                        matrix.emins = nullptr;
                    }
                }
                ASSERT_TRUE(deviceMoEDirectoryCopyReady(source));
                auto runtime = std::make_unique<DeviceMoELayerRuntime>();
                runtime->active_epoch = epoch;
                runtime->expert_count = runtime->top_k = 1u;
                runtime->participant_id = 0u;
                runtime->participant_count = 2u;
                runtime->banks[0].epoch = epoch;
                runtime->banks[0].expert_count = 1u;
                runtime->banks[0].experts[0] = source.descriptor;
                runtime->banks[0].resident_participant_mask[0] = 1u;
                runtime->banks[0].local_compute_mask[0] = 1u;
                DeviceMoERebalancePlanEntry plan;
                plan.op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival);
                plan.source_participant = 0u;
                plan.destination_participant = 1u;
                plan.source_resident_mask = 1u;
                plan.layer = layer;
                plan.destination_slot = slot;
                // Collective packing partitions wire slots by destination. Compact
                // packing must address the same nonzero slot without interpreting
                // it as a directory index.
                plan.payload_slot = 1u;
                plan.destination_generation = observed.generation;
                plan.destination_previous_layer = observed.layer;
                plan.destination_previous_expert = observed.expert;
                DeviceMoERebalanceCommandBufferHeader command;
                command.epoch = epoch;
                command.command_count = command.command_capacity = 1u;
                command.participant_count = 2u;
                auto destination_command = command;
                destination_command.participant_id = 1u;
                const std::array plan_records{plan, plan};
                const std::array command_records{command, destination_command};
                ASSERT_NO_FATAL_FAILURE(upload(source_runtime.as<DeviceMoELayerRuntime>() + layer, runtime.get(), sizeof(*runtime)));
                ASSERT_NO_FATAL_FAILURE(upload(plans.get(), plan_records.data(), sizeof(plan_records)));
                ASSERT_NO_FATAL_FAILURE(upload(header.get(), command_records.data(), sizeof(command_records)));
                ASSERT_NO_FATAL_FAILURE(upload(destination_header.get(), &destination_command, sizeof(destination_command)));
                ASSERT_TRUE(graph->launch());
                DeviceMoERebalanceApplyStatus result;
                ASSERT_NO_FATAL_FAILURE(download(&result, status.get(), sizeof(result)));
                ASSERT_EQ(result.copied_arrivals, 1u);
                ASSERT_EQ(result.descriptor_mismatches, 0u);
                ASSERT_EQ(result.missing_source_descriptors, 0u);
                const auto exact_wire = DeviceMoETransferSlotDirectory::profileForLayerFormats(
                    {specs}, projections).max_wire_payload_bytes;
                EXPECT_EQ(result.copied_payload_bytes, exact_wire)
                    << "Movement evidence must count only the projections actually transmitted";
                ASSERT_NO_FATAL_FAILURE(download(&observed, destination_entry, sizeof(observed)));
                EXPECT_TRUE(deviceMoETransferSlotCopyComplete(observed, 1u, layer, 0u));
                EXPECT_EQ(observed.generation, plan.destination_generation + 1u);
                EXPECT_EQ(observed.descriptor.weight_format, source.descriptor.weight_format);
                EXPECT_EQ(observed.descriptor.projection_set, projections);
                if (projections == DeviceMoEProjectionSet::GateUp)
                {
                    EXPECT_TRUE(observed.descriptor.movableWeightsReady());
                    EXPECT_FALSE(observed.descriptor.weightsReady());
                    EXPECT_EQ(observed.descriptor.down.payload, nullptr);
                    EXPECT_EQ(observed.descriptor.down.scales, nullptr);
                    EXPECT_EQ(observed.descriptor.floating_down.data, nullptr);
                }
                EXPECT_EQ(observed.descriptor.floating_allocation_format, original.floating_allocation_format);
                EXPECT_EQ(observed.descriptor.floating_gate.data, original.floating_gate.data);
                const std::array<DeviceNativeQuantisedMatrixDesc, 3> native_dst{
                    observed.descriptor.gate, observed.descriptor.up, observed.descriptor.down};
                const std::array<DeviceMoEFloatingMatrixDesc, 3> raw_dst{
                    observed.descriptor.floating_gate, observed.descriptor.floating_up,
                    observed.descriptor.floating_down};
                for (size_t projection = 0; projection < specs.size(); ++projection)
                {
                    const auto verify = [&](const void *pointer, size_t count, size_t plane)
                    {
                        if (count == 0u) return;
                        std::vector<uint8_t> bytes(count);
                        ASSERT_NO_FATAL_FAILURE(download(bytes.data(), pointer, count));
                        EXPECT_EQ(std::memcmp(bytes.data(), pattern.data() +
                            (projection * 4u + plane) * plane_bytes, count), 0);
                    };
                    if (specs[projection].format.isFloating())
                    {
                        ASSERT_NO_FATAL_FAILURE(verify(raw_dst[projection].data,
                            deviceMoEFloatingMatrixBytes(raw_dst[projection], observed.descriptor.weight_format), 0));
                    }
                    else
                    {
                        const auto &src = *matrices[projection];
                        const auto &dst = native_dst[projection];
                        if (src.metadata_layout)
                        {
                            namespace sb = native_quantised_superblock;
                            const auto cb = src.codebook_id;
                            const auto blocks = size_t(n) * size_t(k / 32);
                            const auto phase = nativeQuantisedLayoutPhase(src.metadata_layout);
                            const auto code = sb::superblockCodePlaneBits(cb) * blocks;
                            ASSERT_NO_FATAL_FAILURE(verify(dst.payload, blocks * specs[projection].payload_bytes_per_block, 0));
                            ASSERT_NO_FATAL_FAILURE(verify(dst.scales, (code + 7u) / 8u, 1));
                            ASSERT_NO_FATAL_FAILURE(verify(dst.mins, sb::planeBytes(cb, n, k / 32, phase).superblock, 2));
                        }
                        else if (!src.metadata_layout && (src.codebook_id == 11u || src.codebook_id == kNativeQuantisedIQ4XSSourceScaleCodebook))
                        {
                            const size_t blocks = size_t(n) * size_t(k / 32);
                            const size_t payload = blocks * (src.codebook_id == 11u ? 13u : 16u);
                            const size_t coefficient = blocks * sizeof(uint16_t);
                            EXPECT_EQ(deviceMoEMatrixPayloadBytes(src, source), payload);
                            EXPECT_EQ(deviceMoEMatrixScalesBytes(src), coefficient);
                            EXPECT_EQ(deviceMoEMatrixMinsBytes(src, source), coefficient);
                            EXPECT_EQ(deviceMoEMatrixEminsBytes(src, source), 0u);
                            ASSERT_NE(dst.mins, nullptr);
                            ASSERT_NO_FATAL_FAILURE(verify(dst.payload, payload, 0));
                            ASSERT_NO_FATAL_FAILURE(verify(dst.scales, coefficient, 1));
                            ASSERT_NO_FATAL_FAILURE(verify(dst.mins, coefficient, 2));
                        }
                        else
                        {
                            ASSERT_NO_FATAL_FAILURE(verify(dst.payload, deviceMoEMatrixPayloadBytes(src, source), 0));
                            ASSERT_NO_FATAL_FAILURE(verify(dst.scales, deviceMoEMatrixBlockCount(src) * nativeQuantisedScalePlaneBytes(src.codebook_id), 1));
                            ASSERT_NO_FATAL_FAILURE(verify(dst.mins, deviceMoEMatrixMinsBytes(src, source), 2));
                            ASSERT_NO_FATAL_FAILURE(verify(dst.emins, deviceMoEMatrixEminsBytes(src, source), 3));
                        }
                    }
                }
                // A current-generation lease for another family must still fail before mutation.
                if (layer == 0u)
                {
                    auto wrong = plan;
                    wrong.destination_slot = directory->materializedBOM().families[1].range.first;
                    const std::array wrong_plans{wrong, wrong};
                    ASSERT_NO_FATAL_FAILURE(upload(plans.get(), wrong_plans.data(), sizeof(wrong_plans)));
                    ASSERT_TRUE(graph->launch());
                    ASSERT_NO_FATAL_FAILURE(download(&result, status.get(), sizeof(result)));
                    EXPECT_EQ(result.copied_arrivals, 0u);
                    EXPECT_EQ(result.descriptor_mismatches, 1u);
                    ASSERT_NO_FATAL_FAILURE(upload(plans.get(), plan_records.data(), sizeof(plan_records)));
                }
                // Same captured graph and stale lease: reject without changing the
                // installed identity, pointers, dtype, generation or capacity.
                ASSERT_TRUE(graph->launch());
                ASSERT_NO_FATAL_FAILURE(download(&result, status.get(), sizeof(result)));
                EXPECT_EQ(result.copied_arrivals, 0u);
                EXPECT_EQ(result.descriptor_mismatches, 1u);
                DeviceMoEExpertDirectoryEntry after_stale;
                ASSERT_NO_FATAL_FAILURE(download(&after_stale, destination_entry, sizeof(after_stale)));
                EXPECT_EQ(std::memcmp(&after_stale, &observed, sizeof(observed)), 0);
                ASSERT_NO_FATAL_FAILURE(replay_empty_arrival_wave());
            }
        }
    }
} // namespace llaminar2::test
