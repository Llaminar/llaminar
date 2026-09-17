/**
 * @file MixedFormatExpertTransferPublication.cpp
 * @brief Captured CUDA/ROCm transfer-slot reuse across every expert format.
 *
 * The production pack/unpack kernels run in one retained graph. Only setup
 * inputs change between replays; a destination slot keeps its physical address
 * and immutable capacity while its occupant changes. Stale lease replays must
 * leave the previous publication untouched. Arithmetic parity is certified by
 * the separate grouped-expert sweeps; this test isolates the transfer contract.
 */
#include "NativeVNNIExpertTransferParityTest.h"

namespace llaminar2::test
{
    void runMixedFormatExpertTransferPublication(DeviceId device, void *stream)
    {
        using namespace native_vnni_transfer_parity_detail;
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
            // Interleave the widest raw representation with narrower ones, and
            // return to quantized storage afterwards. Capacity cannot follow dtype.
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
            auto directory = DeviceMoETransferSlotDirectory::createForTest(
                backend, device, device.ordinal, 1u, 1u,
                DeviceMoETransferSlotDirectory::profileForLayerFormats(formats));
            ASSERT_NE(directory, nullptr);
            auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
            ASSERT_NE(kernel, nullptr);
            kernel->setGPUStream(stream);
            const MoEKernelLaunchContext launch{.stream = stream};
            DeviceMoERebalanceConfig source_config;
            source_config.num_layers = 1u;
            source_config.num_experts = 1u;
            source_config.top_k = 1u;
            source_config.participant_count = 2u;
            source_config.participant_id = 0u;
            source_config.window_size_tokens = 1u;
            source_config.active_transfer_slot_capacity = 1u;
            source_config.transfer_slot_directory_capacity = 1u;
            auto destination_config = source_config;
            destination_config.participant_id = 1u;
            const size_t payload_bytes = sizeof(DeviceMoEExpertDirectoryEntry) + directory->wirePayloadBytes();
            DeviceAllocation source_weights(backend, device.ordinal, source_bytes);
            DeviceAllocation source_directory(backend, device.ordinal, 2u * sizeof(DeviceMoEExpertDirectoryEntry));
            DeviceAllocation plans(backend, device.ordinal, 2u * sizeof(DeviceMoERebalancePlanEntry));
            DeviceAllocation header(backend, device.ordinal, 2u * sizeof(DeviceMoERebalanceCommandBufferHeader));
            DeviceAllocation destination_header(backend, device.ordinal, sizeof(DeviceMoERebalanceCommandBufferHeader));
            DeviceAllocation payload(backend, device.ordinal, 2u * payload_bytes);
            DeviceAllocation status(backend, device.ordinal, sizeof(DeviceMoERebalanceApplyStatus));
            auto upload = [&](void *dst, const void *src, size_t bytes)
            {
                ASSERT_TRUE(backend->hostToDeviceOnStream(dst, src, bytes, device.ordinal, stream));
            };
            auto download = [&](void *dst, const void *src, size_t bytes)
            {
                ASSERT_TRUE(backend->deviceToHostOnStream(dst, src, bytes, device.ordinal, stream));
                ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
            };
            std::vector<uint8_t> pattern(source_bytes);
            for (size_t i = 0; i < pattern.size(); ++i)
                pattern[i] = static_cast<uint8_t>((i * 53u + i / 257u) & 255u);
            ASSERT_NO_FATAL_FAILURE(upload(source_weights.get(), pattern.data(), pattern.size()));
            DeviceMoEExpertDirectoryEntry observed;
            ASSERT_NO_FATAL_FAILURE(download(&observed, directory->deviceEntries(), sizeof(observed)));
            const auto *immutable_raw_pointer = observed.descriptor.floating_gate.data;
            auto &context = GPUDeviceContextPool::instance().getContext(device);
            auto graph = context.createGraphCapture(stream);
            ASSERT_NE(graph, nullptr);
            ScopedBackendGraphCapture capture(context, *graph, "all-format expert transfer");
            ASSERT_TRUE(capture.begin());
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
                1u, destination_config, status.as<DeviceMoERebalanceApplyStatus>(), nullptr, 1u);
            capture.finish();
            ASSERT_TRUE(packed);
            ASSERT_TRUE(unpacked);
            ASSERT_TRUE(graph->instantiate());

            uint32_t epoch = 0;
            for (const auto &specs : formats)
            {
                SCOPED_TRACE(device.to_string() + " transfer epoch=" + std::to_string(++epoch));
                DeviceMoEExpertDirectoryEntry source;
                source.participant = 0u;
                source.resident_mask = 1u;
                source.slot_index = 0u;
                source.flags = static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Valid) |
                               static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Resident);
                source.descriptor.logical_expert_id = 0;
                source.descriptor.owner_participant = 0;
                source.descriptor.local_slot = 0;
                source.descriptor.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                                           DeviceMoEExpertFlags::Resident);
                const std::array<DeviceNativeVNNIMatrixDesc *, 3> matrices{
                    &source.descriptor.gate, &source.descriptor.up, &source.descriptor.down};
                const std::array<DeviceMoEFloatingMatrixDesc *, 3> raw{
                    &source.descriptor.floating_gate, &source.descriptor.floating_up,
                    &source.descriptor.floating_down};
                for (size_t projection = 0; projection < 3; ++projection)
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
                    auto &matrix = *matrices[projection];
                    matrix.payload = base;
                    matrix.scales = base + plane_bytes;
                    matrix.mins = spec.is_asymmetric ? base + 2u * plane_bytes : nullptr;
                    matrix.emins = spec.has_emins ? base + 3u * plane_bytes : nullptr;
                    matrix.n = n;
                    matrix.k = k;
                    matrix.blocks_per_row = k / 32;
                    matrix.codebook_id = spec.codebook_id;
                    matrix.source_codebook_id = spec.format.native_vnni.codebook_id;
                    matrix.source_is_superblock = spec.format.native_vnni.is_superblock;
                    matrix.source_identity_present = 1u;
                    matrix.allocation_payload_bytes_per_block = spec.payload_bytes_per_block;
                    matrix.allocation_has_mins = spec.is_asymmetric;
                    matrix.allocation_has_emins = spec.has_emins;
                }
                ASSERT_TRUE(deviceMoEDirectoryCopyReady(source));
                std::array<DeviceMoEExpertDirectoryEntry, 2> sources{source, source};
                DeviceMoERebalancePlanEntry plan;
                plan.op = static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival);
                plan.source_participant = 0u;
                plan.destination_participant = 1u;
                plan.source_resident_mask = 1u;
                plan.destination_slot = 0u;
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
                ASSERT_NO_FATAL_FAILURE(upload(source_directory.get(), sources.data(), sizeof(sources)));
                ASSERT_NO_FATAL_FAILURE(upload(plans.get(), plan_records.data(), sizeof(plan_records)));
                ASSERT_NO_FATAL_FAILURE(upload(header.get(), command_records.data(), sizeof(command_records)));
                ASSERT_NO_FATAL_FAILURE(upload(destination_header.get(), &destination_command, sizeof(destination_command)));
                ASSERT_TRUE(graph->launch());
                DeviceMoERebalanceApplyStatus result;
                ASSERT_NO_FATAL_FAILURE(download(&result, status.get(), sizeof(result)));
                ASSERT_EQ(result.copied_arrivals, 1u);
                ASSERT_EQ(result.descriptor_mismatches, 0u);
                ASSERT_EQ(result.missing_source_descriptors, 0u);
                ASSERT_NO_FATAL_FAILURE(download(&observed, directory->deviceEntries(), sizeof(observed)));
                EXPECT_TRUE(deviceMoETransferSlotCopyComplete(observed, 1u, 0u, 0u));
                EXPECT_EQ(observed.generation, plan.destination_generation + 1u);
                EXPECT_EQ(observed.descriptor.weight_format, source.descriptor.weight_format);
                EXPECT_EQ(observed.descriptor.floating_allocation_format, DeviceMoEWeightFormat::FP32);
                EXPECT_EQ(observed.descriptor.floating_gate.data, immutable_raw_pointer);
                const std::array<DeviceNativeVNNIMatrixDesc, 3> native_dst{
                    observed.descriptor.gate, observed.descriptor.up, observed.descriptor.down};
                const std::array<DeviceMoEFloatingMatrixDesc, 3> raw_dst{
                    observed.descriptor.floating_gate, observed.descriptor.floating_up,
                    observed.descriptor.floating_down};
                for (size_t projection = 0; projection < 3; ++projection)
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
                        ASSERT_NO_FATAL_FAILURE(verify(dst.payload, deviceMoEMatrixPayloadBytes(src, source), 0));
                        ASSERT_NO_FATAL_FAILURE(verify(dst.scales, deviceMoEMatrixScalesBytes(src), 1));
                        ASSERT_NO_FATAL_FAILURE(verify(dst.mins, deviceMoEMatrixMinsBytes(src, source), 2));
                        ASSERT_NO_FATAL_FAILURE(verify(dst.emins, deviceMoEMatrixEminsBytes(src, source), 3));
                    }
                }
                // Same captured graph and stale lease: reject without changing the
                // installed identity, pointers, dtype, generation or capacity.
                ASSERT_TRUE(graph->launch());
                ASSERT_NO_FATAL_FAILURE(download(&result, status.get(), sizeof(result)));
                EXPECT_EQ(result.copied_arrivals, 0u);
                EXPECT_EQ(result.descriptor_mismatches, 1u);
                DeviceMoEExpertDirectoryEntry after_stale;
                ASSERT_NO_FATAL_FAILURE(download(&after_stale, directory->deviceEntries(), sizeof(after_stale)));
                EXPECT_EQ(std::memcmp(&after_stale, &observed, sizeof(observed)), 0);
            }
        }
    }
} // namespace llaminar2::test
