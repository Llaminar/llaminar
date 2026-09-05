/**
 * @file DeviceMoETransferSlotDirectory.cpp
 * @brief Persistent participant-local storage for device MoE transfer arrivals.
 *
 * Directory entries separate immutable allocation identity from request-owned
 * logical occupants. Construction validates exact arithmetic format and
 * geometry before publishing pointer-bearing descriptors; request reset then
 * restores the immutable device baseline with one ordered D2D copy.
 */

#include "DeviceMoETransferSlotDirectory.h"

#include "../../backends/IBackend.h"
#include "../../loaders/GPUVramPreflight.h"
#include "../../loaders/gpu_pipeline/LoadOrchestrator.h"
#include "MoEOverlayPhysicalResidencyFabric.h"
#include "../../utils/Logger.h"
#include "../../utils/VramBillOfMaterials.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        DeviceNativeVNNIMatrixDesc descriptorFromSlot(
            const WeightVRAMPool::WeightSlot &slot,
            const DeviceMoETransferSlotDirectory::ProjectionSpec &spec)
        {
            DeviceNativeVNNIMatrixDesc desc;
            desc.payload = slot.d_native_vnni_payload;
            desc.scales = slot.d_native_vnni_scales;
            desc.mins = slot.d_native_vnni_mins;
            desc.emins = slot.d_native_vnni_emins;
            desc.n = spec.N;
            desc.k = spec.K;
            desc.blocks_per_row = static_cast<uint32_t>(spec.K / 32);
            desc.codebook_id = spec.codebook_id;
            desc.allocation_payload_bytes_per_block =
                static_cast<uint8_t>(spec.payload_bytes_per_block);
            desc.allocation_has_mins = spec.is_asymmetric ? 1u : 0u;
            desc.allocation_has_emins = spec.has_emins ? 1u : 0u;
            if (!spec.format.isNativeVnni())
                return {};
            desc.source_codebook_id = spec.format.native_vnni.codebook_id;
            desc.source_is_superblock = static_cast<uint8_t>(
                spec.format.native_vnni.is_superblock);
            desc.source_identity_present = static_cast<uint8_t>(
                spec.format.native_vnni.present);
            return desc;
        }

        void setProjectionDescriptor(
            DeviceMoEExpertDescriptor &expert_desc,
            const std::string &label,
            const DeviceNativeVNNIMatrixDesc &matrix_desc)
        {
            if (label == "gate")
                expert_desc.gate = matrix_desc;
            else if (label == "up")
                expert_desc.up = matrix_desc;
            else if (label == "down")
                expert_desc.down = matrix_desc;
            else
                throw std::invalid_argument(
                    "DeviceMoETransferSlotDirectory projection label must be gate, up, or down");
        }

        void validateSpecs(
            const std::vector<DeviceMoETransferSlotDirectory::ProjectionSpec> &specs)
        {
            if (specs.size() != 3)
                throw std::invalid_argument("DeviceMoETransferSlotDirectory requires gate/up/down specs");

            bool has_gate = false;
            bool has_up = false;
            bool has_down = false;
            for (const auto &spec : specs)
            {
                if (spec.label == "gate")
                    has_gate = true;
                else if (spec.label == "up")
                    has_up = true;
                else if (spec.label == "down")
                    has_down = true;
                else
                    throw std::invalid_argument(
                        "DeviceMoETransferSlotDirectory projection label must be gate, up, or down");

                if (spec.N <= 0 || spec.K <= 0 || (spec.K % 32) != 0 ||
                    spec.payload_bytes_per_block <= 0 ||
                    !spec.format.valid())
                {
                    throw std::invalid_argument(
                        "DeviceMoETransferSlotDirectory projection spec has invalid NativeVNNI shape or source format");
                }
            }

            if (!has_gate || !has_up || !has_down)
                throw std::invalid_argument("DeviceMoETransferSlotDirectory requires gate/up/down specs");
        }

        size_t projectionPayloadBytes(
            const DeviceMoETransferSlotDirectory::ProjectionSpec &spec)
        {
            const size_t block_count =
                static_cast<size_t>(spec.N) * static_cast<size_t>(spec.K / 32);
            const size_t scales_bytes = block_count * sizeof(uint16_t);
            return block_count * static_cast<size_t>(spec.payload_bytes_per_block) +
                   scales_bytes +
                   (spec.is_asymmetric ? scales_bytes : 0u) +
                   (spec.has_emins ? block_count * sizeof(uint32_t) : 0u);
        }

        size_t expertPayloadBytes(
            const std::vector<DeviceMoETransferSlotDirectory::ProjectionSpec> &specs)
        {
            size_t bytes = 0;
            for (const auto &spec : specs)
                bytes += projectionPayloadBytes(spec);
            return bytes;
        }

        /** @return Stable key shared by pure planning and live allocation. */
        std::string directorySlotName(
            uint32_t slot_index,
            const std::string &label)
        {
            return "moe_device_rebalance_transfer_slot_" +
                   std::to_string(slot_index) + "_" + label;
        }

        /** @return Allocator label for one typed model-manifest projection. */
        const char *projectionLabel(ExpertTierWeightProjection projection)
        {
            switch (projection)
            {
            case ExpertTierWeightProjection::Gate:
                return "gate";
            case ExpertTierWeightProjection::Up:
                return "up";
            case ExpertTierWeightProjection::Down:
                return "down";
            }
            throw std::invalid_argument(
                "DeviceMoETransferSlotDirectory received an unknown projection role");
        }

        /** @brief Submit every slot region to the allocator-owned byte planner. */
        void planDirectoryPayload(
            WeightVRAMPool &pool,
            uint32_t slot_count,
            const std::vector<DeviceMoETransferSlotDirectory::ProjectionSpec>
                &specs)
        {
            for (uint32_t slot = 0; slot < slot_count; ++slot)
            {
                for (const auto &spec : specs)
                {
                    pool.planWeight(
                        directorySlotName(slot, spec.label),
                        spec.N,
                        spec.K,
                        spec.payload_bytes_per_block,
                        spec.is_asymmetric,
                        spec.has_emins,
                        /*raw_gguf_bytes=*/0);
                }
            }
        }
    } // namespace

    DeviceMoETransferSlotDirectory::FormatProfile
    DeviceMoETransferSlotDirectory::profileForLayerFormats(
        const std::vector<std::vector<ProjectionSpec>> &layer_formats)
    {
        if (layer_formats.empty())
        {
            throw std::invalid_argument(
                "DeviceMoETransferSlotDirectory requires at least one layer format");
        }

        validateSpecs(layer_formats.front());
        FormatProfile profile;
        profile.allocation_specs = layer_formats.front();
        profile.max_wire_payload_bytes = expertPayloadBytes(layer_formats.front());

        for (size_t layer = 1; layer < layer_formats.size(); ++layer)
        {
            const auto &layer_specs = layer_formats[layer];
            validateSpecs(layer_specs);
            profile.max_wire_payload_bytes =
                std::max(profile.max_wire_payload_bytes, expertPayloadBytes(layer_specs));

            for (auto &allocation : profile.allocation_specs)
            {
                const auto source_it = std::find_if(
                    layer_specs.begin(),
                    layer_specs.end(),
                    [&](const ProjectionSpec &candidate)
                    {
                        return candidate.label == allocation.label;
                    });
                if (source_it == layer_specs.end() ||
                    source_it->N != allocation.N ||
                    source_it->K != allocation.K)
                {
                    throw std::invalid_argument(
                        "DeviceMoETransferSlotDirectory layer formats have incompatible " +
                        allocation.label + " projection geometry");
                }

                /*
                 * These are allocation capacities, not a synthetic codebook.
                 * Keep the first observed codebook only as valid initial
                 * descriptor metadata; the unpack kernel retargets it before
                 * publishing each arrival.
                 */
                allocation.payload_bytes_per_block =
                    std::max(
                        allocation.payload_bytes_per_block,
                        source_it->payload_bytes_per_block);
                allocation.is_asymmetric =
                    allocation.is_asymmetric || source_it->is_asymmetric;
                allocation.has_emins =
                    allocation.has_emins || source_it->has_emins;
            }
        }
        return profile;
    }

    DeviceMoETransferSlotDirectory::FormatProfile
    DeviceMoETransferSlotDirectory::profileForLayerWeightManifest(
        const std::vector<MoEOverlayLayerWeightManifest> &
            layer_weight_manifest)
    {
        if (layer_weight_manifest.empty())
        {
            throw std::invalid_argument(
                "DeviceMoETransferSlotDirectory requires a non-empty model manifest");
        }

        std::vector<std::vector<ProjectionSpec>> layer_formats;
        layer_formats.reserve(layer_weight_manifest.size());
        for (std::size_t layer_offset = 0;
             layer_offset < layer_weight_manifest.size();
             ++layer_offset)
        {
            const auto &layer = layer_weight_manifest[layer_offset];
            if (!layer.valid() ||
                layer.layer_idx != static_cast<int>(layer_offset))
            {
                throw std::invalid_argument(
                    "DeviceMoETransferSlotDirectory manifest must be valid and contiguous from layer zero");
            }

            std::vector<ProjectionSpec> specs;
            specs.reserve(layer.projections.size());
            for (const auto &projection : layer.projections)
            {
                if (!projection.format.isNativeVnni())
                {
                    throw std::invalid_argument(
                        "DeviceMoETransferSlotDirectory currently requires NativeVNNI expert weights");
                }
                const auto *source =
                    native_vnni_formats::forSourceIdentity(
                        projection.format.native_vnni.codebook_id,
                        projection.format.native_vnni.is_superblock);
                if (!source)
                {
                    throw std::invalid_argument(
                        "DeviceMoETransferSlotDirectory manifest contains an uncatalogued NativeVNNI source format");
                }

                /*
                 * The prepared engine can later carry either its initial
                 * compact bytes or a migration-stable representation. Its
                 * exported allocation fields describe that union, so
                 * setup-time admission must derive the identical capacity.
                 */
                const auto reusable =
                    reusableDeviceVnniAllocationFormat(*source);
                specs.push_back({
                    .label = projectionLabel(projection.projection),
                    .N = projection.N,
                    .K = projection.K,
                    .payload_bytes_per_block =
                        static_cast<int>(
                            reusable.payload_bytes_per_block),
                    .is_asymmetric = reusable.has_mins,
                    .has_emins = reusable.has_emins,
                    .codebook_id =
                        canonicalDeviceVnniCodebookId(
                            source->codebook_id),
                    .format = projection.format,
                });
            }
            layer_formats.push_back(std::move(specs));
        }
        return profileForLayerFormats(layer_formats);
    }

    uint64_t DeviceMoETransferSlotDirectory::persistentActiveSlotDemand(
        const DeviceMoERebalanceConfig &config) noexcept
    {
        if (config.num_layers == 0u)
            return 0u;

        const uint64_t active_lanes_per_layer =
            static_cast<uint64_t>(
                std::max<uint32_t>(
                    1u,
                    config.max_hot_replicas_per_participant));
        return static_cast<uint64_t>(config.num_layers) *
               active_lanes_per_layer;
    }

    DeviceMoETransferSlotDirectory::BufferedCapacity
    DeviceMoETransferSlotDirectory::planBufferedCapacity(
        uint64_t requested_active_slots,
        uint32_t transfer_wave_slots,
        uint32_t transfer_buffer_count)
    {
        if (requested_active_slots == 0)
        {
            throw std::invalid_argument(
                "DeviceMoETransferSlotDirectory active slot demand must be nonzero");
        }
        if (transfer_wave_slots == 0)
        {
            throw std::invalid_argument(
                "DeviceMoETransferSlotDirectory transfer wave capacity must be nonzero");
        }
        if (transfer_buffer_count == 0)
        {
            throw std::invalid_argument(
                "DeviceMoETransferSlotDirectory transfer buffer count must be nonzero");
        }

        const uint64_t staging_slots =
            static_cast<uint64_t>(transfer_wave_slots) *
            static_cast<uint64_t>(transfer_buffer_count);
        const uint64_t total_slots = requested_active_slots + staging_slots;
        if (requested_active_slots >
                static_cast<uint64_t>(kDeviceMoEMaxTransferSlots) ||
            staging_slots >
                static_cast<uint64_t>(kDeviceMoEMaxTransferSlots) ||
            total_slots >
                static_cast<uint64_t>(kDeviceMoEMaxTransferSlots))
        {
            throw std::invalid_argument(
                "DeviceMoETransferSlotDirectory buffered capacity exceeds the "
                "device transfer-slot address space: active=" +
                std::to_string(requested_active_slots) +
                " staging=" + std::to_string(staging_slots) +
                " total=" + std::to_string(total_slots) +
                " maximum=" +
                std::to_string(kDeviceMoEMaxTransferSlots));
        }

        return BufferedCapacity{
            .active_slots = static_cast<uint32_t>(requested_active_slots),
            .staging_slots = static_cast<uint32_t>(staging_slots),
            .total_slots = static_cast<uint32_t>(total_slots),
        };
    }

    DeviceMoETransferSlotDirectory::BufferedCapacity
    DeviceMoETransferSlotDirectory::planRuntimeCapacity(
        const DeviceMoERebalanceConfig &config,
        uint64_t minimum_active_slots,
        uint32_t transfer_wave_slots,
        uint32_t transfer_buffer_count)
    {
        return planBufferedCapacity(
            std::max(
                persistentActiveSlotDemand(config),
                minimum_active_slots),
            transfer_wave_slots,
            transfer_buffer_count);
    }

    DeviceMoETransferSlotDirectory::AllocationBOM
    DeviceMoETransferSlotDirectory::allocationBOM(
        BufferedCapacity capacity,
        const FormatProfile &format_profile)
    {
        if (capacity.active_slots == 0u ||
            capacity.staging_slots == 0u ||
            capacity.total_slots == 0u ||
            static_cast<uint64_t>(capacity.active_slots) +
                    static_cast<uint64_t>(capacity.staging_slots) !=
                capacity.total_slots)
        {
            throw std::invalid_argument(
                "DeviceMoETransferSlotDirectory allocation BOM requires a coherent active/staging capacity");
        }
        validateSpecs(format_profile.allocation_specs);
        if (format_profile.max_wire_payload_bytes == 0u ||
            format_profile.max_wire_payload_bytes >
                expertPayloadBytes(format_profile.allocation_specs))
        {
            throw std::invalid_argument(
                "DeviceMoETransferSlotDirectory allocation BOM has an invalid wire payload capacity");
        }

        WeightVRAMPool planner;
        planDirectoryPayload(
            planner,
            capacity.total_slots,
            format_profile.allocation_specs);
        const size_t payload_bytes = alignGPUWeightLoadAllocation(
            planner.totalPlannedBytes());
        if (capacity.total_slots >
            std::numeric_limits<size_t>::max() /
                sizeof(DeviceMoEExpertDirectoryEntry) / 2u)
        {
            throw std::overflow_error(
                "DeviceMoETransferSlotDirectory descriptor BOM overflows size_t");
        }
        const size_t descriptor_bytes =
            static_cast<size_t>(capacity.total_slots) *
            sizeof(DeviceMoEExpertDirectoryEntry) * 2u;
        if (descriptor_bytes >
            std::numeric_limits<size_t>::max() - payload_bytes)
        {
            throw std::overflow_error(
                "DeviceMoETransferSlotDirectory complete BOM overflows size_t");
        }
        return {
            .payload_bytes = payload_bytes,
            .descriptor_bytes = descriptor_bytes,
            .total_bytes = payload_bytes + descriptor_bytes,
        };
    }

    std::shared_ptr<DeviceMoETransferSlotDirectory>
    DeviceMoETransferSlotDirectory::create(
        IBackend *backend,
        DeviceId device,
        int device_ordinal,
        uint32_t participant_id,
        BufferedCapacity capacity,
        FormatProfile format_profile,
        std::shared_ptr<PhysicalMemoryAuthority> memory_authority)
    {
        return createImpl(
            backend,
            device,
            device_ordinal,
            participant_id,
            capacity,
            std::move(format_profile),
            std::move(memory_authority),
            /*explicit_test_allocation=*/false);
    }

    std::shared_ptr<DeviceMoETransferSlotDirectory>
    DeviceMoETransferSlotDirectory::createForTest(
        IBackend *backend,
        DeviceId device,
        int device_ordinal,
        uint32_t participant_id,
        uint32_t slot_count,
        FormatProfile format_profile)
    {
        return createImpl(
            backend,
            device,
            device_ordinal,
            participant_id,
            BufferedCapacity{
                .active_slots = slot_count,
                .staging_slots = 0u,
                .total_slots = slot_count,
            },
            std::move(format_profile),
            nullptr,
            /*explicit_test_allocation=*/true);
    }

    std::shared_ptr<DeviceMoETransferSlotDirectory>
    DeviceMoETransferSlotDirectory::createImpl(
        IBackend *backend,
        DeviceId device,
        int device_ordinal,
        uint32_t participant_id,
        BufferedCapacity capacity,
        FormatProfile format_profile,
        std::shared_ptr<PhysicalMemoryAuthority> memory_authority,
        bool explicit_test_allocation)
    {
        if (!backend)
            throw std::invalid_argument("DeviceMoETransferSlotDirectory requires a backend");
        if (!device.is_gpu())
            throw std::invalid_argument("DeviceMoETransferSlotDirectory requires a GPU device");
        if (device_ordinal < 0)
            throw std::invalid_argument("DeviceMoETransferSlotDirectory requires a GPU device ordinal");
        if (device.gpu_ordinal() != device_ordinal)
        {
            throw std::invalid_argument(
                "DeviceMoETransferSlotDirectory DeviceId/allocator ordinal mismatch: device=" +
                device.to_string() + " allocator_ordinal=" +
                std::to_string(device_ordinal));
        }
        const uint32_t slot_count = capacity.total_slots;
        if (slot_count == 0)
            throw std::invalid_argument("DeviceMoETransferSlotDirectory requires at least one slot");
        if (!explicit_test_allocation &&
            (capacity.active_slots == 0u || capacity.staging_slots == 0u ||
             static_cast<uint64_t>(capacity.active_slots) +
                     static_cast<uint64_t>(capacity.staging_slots) !=
                 capacity.total_slots))
        {
            throw std::invalid_argument(
                "DeviceMoETransferSlotDirectory production allocation requires coherent active/staging capacity");
        }
        if (!explicit_test_allocation && !memory_authority)
        {
            throw std::invalid_argument(
                "DeviceMoETransferSlotDirectory production allocation requires a physical-memory authority");
        }
        auto &specs = format_profile.allocation_specs;
        validateSpecs(specs);
        const size_t slot_storage_capacity_bytes = expertPayloadBytes(specs);
        if (format_profile.max_wire_payload_bytes == 0 ||
            format_profile.max_wire_payload_bytes > slot_storage_capacity_bytes)
        {
            throw std::invalid_argument(
                "DeviceMoETransferSlotDirectory wire payload exceeds allocation capacity");
        }

        auto orchestrator = explicit_test_allocation
                                ? std::make_shared<LoadOrchestrator>(
                                      backend,
                                      kTestOnlyUnadmittedGPUAllocation)
                                : std::make_shared<LoadOrchestrator>(
                                      backend,
                                      memory_authority,
                                      PhysicalMemoryOwner::ExpertMigrationStaging);
        orchestrator->addDevice(device_ordinal);

        for (uint32_t slot = 0; slot < slot_count; ++slot)
            for (const auto &spec : specs)
                orchestrator->planWeight(
                    device_ordinal,
                    slotName(slot, spec.label),
                    spec.N,
                    spec.K,
                    spec.payload_bytes_per_block,
                    spec.is_asymmetric,
                    spec.has_emins,
                    /*raw_gguf_bytes=*/0);
        orchestrator->allocate(/*pinned_slot_size=*/0, /*num_h2d_streams=*/0);

        const auto *weight_pool = orchestrator->getPool(device_ordinal);
        if (!weight_pool || !weight_pool->isAllocated())
            throw std::runtime_error("DeviceMoETransferSlotDirectory failed to allocate slot weights");
        const size_t planned_bytes = weight_pool->totalPlannedBytes();
        if (!explicit_test_allocation)
        {
            const auto expected = allocationBOM(capacity, format_profile);
            if (planned_bytes != expected.payload_bytes)
            {
                throw std::logic_error(
                    "DeviceMoETransferSlotDirectory allocator bytes diverged from setup admission");
            }
        }

        std::vector<DeviceMoEExpertDirectoryEntry> host_entries(slot_count);
        for (uint32_t slot = 0; slot < slot_count; ++slot)
        {
            DeviceMoEExpertDirectoryEntry entry;
            entry.layer = kDeviceMoEInvalidSlot;
            entry.expert = kDeviceMoEInvalidSlot;
            entry.participant = participant_id;
            entry.slot_index = slot;
            entry.descriptor.logical_expert_id = -1;
            entry.descriptor.owner_participant = static_cast<int32_t>(participant_id);
            entry.descriptor.local_slot = static_cast<int32_t>(slot);
            entry.descriptor.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                                      DeviceMoEExpertFlags::TransferSlot);
            entry.flags =
                static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Valid) |
                static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::TransferSlot);

            for (const auto &spec : specs)
            {
                auto maybe_slot = weight_pool->getSlot(slotName(slot, spec.label));
                if (!maybe_slot.has_value())
                    throw std::runtime_error("DeviceMoETransferSlotDirectory missing planned transfer slot");
                setProjectionDescriptor(
                    entry.descriptor,
                    spec.label,
                    descriptorFromSlot(*maybe_slot, spec));
            }
            if (!deviceMoETransferSlotReadyForCopy(
                    entry,
                    participant_id,
                    /*layer=*/0,
                    /*expert=*/0))
            {
                throw std::runtime_error(
                    "DeviceMoETransferSlotDirectory produced an unusable generic transfer slot");
            }
            host_entries[slot] = entry;
        }

        if (host_entries.size() >
            std::numeric_limits<size_t>::max() /
                sizeof(host_entries[0]))
        {
            throw std::overflow_error(
                "DeviceMoETransferSlotDirectory descriptor byte count overflows size_t");
        }
        const size_t directory_bytes =
            host_entries.size() * sizeof(host_entries[0]);
        if (directory_bytes > std::numeric_limits<size_t>::max() / 2u)
        {
            throw std::overflow_error(
                "DeviceMoETransferSlotDirectory baseline byte count overflows size_t");
        }
        std::optional<PhysicalMemoryAllocationLease>
            directory_entries_lease;
        if (!explicit_test_allocation)
        {
            directory_entries_lease.emplace(
                memory_authority->claimNewAllocation(
                    device,
                    PhysicalMemoryOwner::ExpertMigrationStaging,
                    directory_bytes * 2u));
        }

        DeviceMoEExpertDirectoryEntry *device_entries =
            static_cast<DeviceMoEExpertDirectoryEntry *>(
                backend->allocate(directory_bytes, device_ordinal));
        if (!device_entries)
            throw std::runtime_error("DeviceMoETransferSlotDirectory failed to allocate device directory");
        DeviceMoEExpertDirectoryEntry *device_baseline_entries =
            static_cast<DeviceMoEExpertDirectoryEntry *>(
                backend->allocate(directory_bytes, device_ordinal));
        if (!device_baseline_entries)
        {
            backend->free(device_entries, device_ordinal);
            throw std::runtime_error(
                "DeviceMoETransferSlotDirectory failed to allocate device baseline directory");
        }

        void *upload_stream = backend->createStream(device_ordinal);
        if (!upload_stream)
        {
            backend->free(device_baseline_entries, device_ordinal);
            backend->free(device_entries, device_ordinal);
            throw std::runtime_error("DeviceMoETransferSlotDirectory failed to create upload stream");
        }

        const bool uploaded =
            backend->hostToDeviceOnStream(
                device_entries,
                host_entries.data(),
                directory_bytes,
                device_ordinal,
                upload_stream) &&
            backend->deviceCopyAsync(
                device_baseline_entries,
                device_entries,
                directory_bytes,
                device_ordinal,
                upload_stream) &&
            backend->synchronizeStream(upload_stream, device_ordinal);
        backend->destroyStream(upload_stream, device_ordinal);
        if (!uploaded)
        {
            backend->free(device_baseline_entries, device_ordinal);
            backend->free(device_entries, device_ordinal);
            throw std::runtime_error("DeviceMoETransferSlotDirectory failed to upload device directory");
        }

        auto directory = std::shared_ptr<DeviceMoETransferSlotDirectory>(
            new DeviceMoETransferSlotDirectory(
                backend,
                device,
                device_ordinal,
                participant_id,
                capacity,
                std::move(format_profile.allocation_specs),
                std::move(orchestrator),
                device_entries,
                device_baseline_entries,
                std::move(host_entries),
                std::move(directory_entries_lease),
                planned_bytes,
                format_profile.max_wire_payload_bytes,
                slot_storage_capacity_bytes));

        logVramBomLine(
            "moe_device_transfer_slot_directory",
            "backend=" + std::string(backend->backendName()) +
                " device=" + device.to_string() +
                " device_id=" + std::to_string(device_ordinal) +
                " participant=" + std::to_string(participant_id) +
                " slots=" + std::to_string(slot_count) +
                " active_slots=" +
                std::to_string(capacity.active_slots) +
                " staging_slots=" +
                std::to_string(capacity.staging_slots) +
                " projections_per_slot=" + std::to_string(directory->specs_.size()) +
                " collective_slot_payload_bytes=" + std::to_string(directory->wire_payload_bytes_) +
                " slot_storage_capacity_bytes=" + std::to_string(slot_storage_capacity_bytes) +
                " total_planned_bytes=" + std::to_string(planned_bytes) +
                " total_planned_mib=" + vramBomMiB(planned_bytes) +
                " directory_bytes=" + std::to_string(directory_bytes) +
                " directory_mib=" + vramBomMiB(directory_bytes));
        return directory;
    }

    DeviceMoETransferSlotDirectory::DeviceMoETransferSlotDirectory(
        IBackend *backend,
        DeviceId device,
        int device_ordinal,
        uint32_t participant_id,
        BufferedCapacity capacity,
        std::vector<ProjectionSpec> specs,
        std::shared_ptr<LoadOrchestrator> orchestrator,
        DeviceMoEExpertDirectoryEntry *device_entries,
        DeviceMoEExpertDirectoryEntry *device_baseline_entries,
        std::vector<DeviceMoEExpertDirectoryEntry> host_entries,
        std::optional<PhysicalMemoryAllocationLease>
            directory_entries_lease,
        size_t planned_bytes,
        size_t wire_payload_bytes,
        size_t slot_storage_capacity_bytes)
        : backend_(backend),
          device_(device),
          device_ordinal_(device_ordinal),
          participant_id_(participant_id),
          slot_count_(capacity.total_slots),
          capacity_(capacity),
          specs_(std::move(specs)),
          orchestrator_(std::move(orchestrator)),
          device_entries_(device_entries),
          device_baseline_entries_(device_baseline_entries),
          host_entries_(std::move(host_entries)),
          directory_entries_lease_(std::move(directory_entries_lease)),
          planned_bytes_(planned_bytes),
          wire_payload_bytes_(wire_payload_bytes),
          slot_storage_capacity_bytes_(slot_storage_capacity_bytes)
    {
    }

    DeviceMoETransferSlotDirectory::~DeviceMoETransferSlotDirectory()
    {
        if (device_baseline_entries_ && backend_)
        {
            backend_->free(device_baseline_entries_, device_ordinal_);
            device_baseline_entries_ = nullptr;
        }
        if (device_entries_ && backend_)
        {
            backend_->free(device_entries_, device_ordinal_);
            device_entries_ = nullptr;
        }
    }

    void DeviceMoETransferSlotDirectory::resetRequestPublications(
        void *stream)
    {
        if (!stream)
        {
            throw std::invalid_argument(
                "DeviceMoETransferSlotDirectory request reset requires an explicit stream");
        }
        if (!backend_ || !device_entries_ || !device_baseline_entries_ ||
            slot_count_ == 0u)
        {
            throw std::logic_error(
                "DeviceMoETransferSlotDirectory request reset has incomplete model-lifetime ownership");
        }

        const size_t bytes =
            static_cast<size_t>(slot_count_) *
            sizeof(DeviceMoEExpertDirectoryEntry);
        /*
         * The live directory and its immutable baseline are both owned by this
         * GPU.  Enqueue the reset behind the caller's already-ordered request
         * boundary work.  The next graph on this stream therefore observes the
         * restored directory without blocking the host or introducing a
         * separate coherence state.
         */
        if (!backend_->deviceCopyAsync(
                device_entries_,
                device_baseline_entries_,
                bytes,
                device_ordinal_,
                stream))
        {
            throw std::runtime_error(
                "DeviceMoETransferSlotDirectory failed to enqueue request-publication reset on " +
                device_.to_string());
        }
    }

    void DeviceMoETransferSlotDirectory::requirePhysicalOwner(
        DeviceId expected_device,
        int expected_device_ordinal,
        uint32_t expected_participant_id) const
    {
        if (device_ == expected_device &&
            device_ordinal_ == expected_device_ordinal &&
            participant_id_ == expected_participant_id)
        {
            return;
        }

        throw std::logic_error(
            "DeviceMoETransferSlotDirectory physical-owner mismatch: actual_device=" +
            device_.to_string() + " actual_ordinal=" +
            std::to_string(device_ordinal_) + " actual_participant=" +
            std::to_string(participant_id_) + " expected_device=" +
            expected_device.to_string() + " expected_ordinal=" +
            std::to_string(expected_device_ordinal) +
            " expected_participant=" +
            std::to_string(expected_participant_id));
    }

    bool DeviceMoETransferSlotDirectory::descriptorForSlot(
        uint32_t slot_index,
        uint32_t logical_expert,
        DeviceMoEExpertDescriptor &out) const
    {
        if (slot_index >= host_entries_.size())
            return false;

        const auto &entry = host_entries_[static_cast<size_t>(slot_index)];
        if (!deviceMoETransferSlotReadyForCopy(
                entry,
                participant_id_,
                /*layer=*/0,
                /*expert=*/0))
        {
            return false;
        }

        out = entry.descriptor;
        out.logical_expert_id = static_cast<int32_t>(logical_expert);
        out.local_slot = static_cast<int32_t>(slot_index);
        out.flags |= toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                      DeviceMoEExpertFlags::Resident |
                                      DeviceMoEExpertFlags::TransferSlot);
        return true;
    }

    std::string DeviceMoETransferSlotDirectory::slotName(
        uint32_t slot_index,
        const std::string &label)
    {
        return directorySlotName(slot_index, label);
    }

} // namespace llaminar2
