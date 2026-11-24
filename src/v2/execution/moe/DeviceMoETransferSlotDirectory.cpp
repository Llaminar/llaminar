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
#include "GPUExpertProjectionContract.h"
#include "execution/config/RuntimeConfig.h"
#include "loaders/GPUPreparedWeightPoolLayout.h"

#include "../../backends/IBackend.h"
#include "../../loaders/GPUVramPreflight.h"
#include "MoEOverlayPhysicalResidencyFabric.h"
#include "../../utils/Logger.h"
#include "../../utils/VramBillOfMaterials.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** @return Device arithmetic tag for an authenticated source format. */
        DeviceMoEWeightFormat deviceFormat(const ExpertWeightFormat &format)
        {
            switch (format.kind)
            {
            case ExpertWeightFormatKind::NativeQuantised: return DeviceMoEWeightFormat::NativeQuantised;
            case ExpertWeightFormatKind::FP16: return DeviceMoEWeightFormat::FP16;
            case ExpertWeightFormatKind::BF16: return DeviceMoEWeightFormat::BF16;
            case ExpertWeightFormatKind::FP32: return DeviceMoEWeightFormat::FP32;
            default: throw std::invalid_argument("Transfer directory requires a valid weight format");
            }
        }

        /** @return Checked logical raw bytes; the allocator owns physical padding. */
        size_t floatingBytes(
            const DeviceMoETransferSlotDirectory::ProjectionSpec &spec,
            DeviceMoEWeightFormat format)
        {
            return deviceMoEFloatingMatrixBytes({.n = spec.N, .k = spec.K}, format);
        }

        /** @return Native plane view, absent when no quantized storage was planned. */
        DeviceNativeQuantisedMatrixDesc descriptorFromSlot(
            const WeightVRAMPool::WeightSlot &slot,
            const DeviceMoETransferSlotDirectory::ProjectionSpec &spec)
        {
            DeviceNativeQuantisedMatrixDesc desc;
            desc.metadata_layout = slot.metadata_layout;
            desc.payload = slot.d_native_quantised_payload;
            desc.scales = slot.metadata_layout && native_quantised_superblock::superblockCodeInPayload(spec.codebook_id)
                ? desc.payload : slot.d_native_quantised_scales;
            desc.mins = slot.d_native_quantised_mins;
            desc.emins = slot.d_native_quantised_emins;
            desc.n = spec.N;
            desc.k = spec.K;
            desc.blocks_per_row = static_cast<uint32_t>(spec.K / 32);
            desc.codebook_id = spec.codebook_id;
            desc.allocation_payload_bytes_per_block =
                static_cast<uint8_t>(spec.payload_bytes_per_block);
            desc.allocation_has_mins = spec.is_asymmetric ? 1u : 0u;
            desc.allocation_has_emins = spec.has_emins ? 1u : 0u;
            if (!spec.format.isNativeQuantised())
                return {};
            desc.source_codebook_id = spec.format.native_quantised.codebook_id;
            desc.source_is_superblock = static_cast<uint8_t>(
                spec.format.native_quantised.is_superblock);
            desc.source_identity_present = static_cast<uint8_t>(
                spec.format.native_quantised.present);
            return desc;
        }

        /** @brief Attach the one admitted storage representation to a stable expert slot. */
        void setProjectionDescriptor(
            DeviceMoEExpertDescriptor &expert_desc,
            const DeviceMoETransferSlotDirectory::ProjectionSpec &spec,
            const WeightVRAMPool::WeightSlot &slot,
            DeviceMoEWeightFormat floating_capacity)
        {
            const auto matrix_desc = descriptorFromSlot(slot, spec);
            const DeviceMoEFloatingMatrixDesc floating_desc =
                deviceMoEWeightFormatIsFloating(floating_capacity)
                    ? DeviceMoEFloatingMatrixDesc{slot.d_native_quantised_payload, spec.N, spec.K}
                    : DeviceMoEFloatingMatrixDesc{};
            if (spec.label == "gate")
            {
                expert_desc.gate = matrix_desc;
                expert_desc.floating_gate = floating_desc;
            }
            else if (spec.label == "up")
            {
                expert_desc.up = matrix_desc;
                expert_desc.floating_up = floating_desc;
            }
            else if (spec.label == "down")
            {
                expert_desc.down = matrix_desc;
                expert_desc.floating_down = floating_desc;
            }
            else
                throw std::invalid_argument(
                    "DeviceMoETransferSlotDirectory projection label must be gate, up, or down");
        }

        /** @return Checkpoint format and arithmetic family for a diagnostic only. */
        std::string projectionFormatName(const ExpertWeightFormat &format)
        {
            if (format.isNativeQuantised())
            {
                const auto *source = native_quantised_formats::forSourceIdentity(
                    format.native_quantised.codebook_id, format.native_quantised.is_superblock);
                const auto entry = std::find_if(native_quantised_formats::kAllSourceFormats.begin(),
                    native_quantised_formats::kAllSourceFormats.end(),
                    [&](const auto &candidate) { return candidate.metadata == source; });
                return entry != native_quantised_formats::kAllSourceFormats.end()
                    ? "NativeQuantised/" + std::string(entry->quant_type)
                    : "invalid NativeQuantised source";
            }
            switch (format.kind)
            {
            case ExpertWeightFormatKind::FP16: return "floating/FP16";
            case ExpertWeightFormatKind::BF16: return "floating/BF16";
            case ExpertWeightFormatKind::FP32: return "floating/FP32";
            default: return "invalid";
            }
        }

        /**
         * @brief Require exactly the supported movable family before allocation.
         * @param specs Authenticated movable projections, not the fixed down bank.
         * @param projections Frozen physical ownership contract.
         * @param layer_index Model layer for metadata-derived diagnostics, or -1
         *        for a standalone directory whose caller did not supply a layer.
         * @throws std::invalid_argument For malformed geometry or a mixed family.
         *
         * The compact device descriptor carries one arithmetic tag for the
         * movable payload. Dropping this check would reinterpret floating bytes
         * as NativeQuantised planes. Name every source tensor on rejection so the user
         * can distinguish that implementation limit from a capacity failure.
         */
        void validateSpecs(
            const std::vector<DeviceMoETransferSlotDirectory::ProjectionSpec> &specs,
            DeviceMoEProjectionSet projections,
            int layer_index = -1)
        {
            const bool requires_down = deviceMoEProjectionSetHasDown(projections);
            if (!deviceMoEProjectionSetValid(projections) || specs.size() != (requires_down ? 3u : 2u))
                throw std::invalid_argument("DeviceMoETransferSlotDirectory requires exactly its movable projection set");

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

                if (spec.N <= 0 || spec.K <= 0 || !spec.format.valid() ||
                    (spec.format.isNativeQuantised() &&
                     ((spec.K % 32) != 0 || spec.payload_bytes_per_block <= 0)))
                {
                    throw std::invalid_argument(
                        "DeviceMoETransferSlotDirectory " + spec.label +
                        " projection spec has invalid geometry or source format");
                }
            }

            if (!has_gate || !has_up || has_down != requires_down)
                throw std::invalid_argument("DeviceMoETransferSlotDirectory projection labels differ from its ownership contract");

            const bool mixed = std::any_of(specs.begin(), specs.end(), [&](const auto &spec) {
                return spec.format.isNativeQuantised() != specs.front().format.isNativeQuantised() ||
                       (spec.format.isFloating() && spec.format != specs.front().format);
            });
            if (mixed)
            {
                std::ostringstream message;
                message << "ExpertOverlay movable ";
                if (layer_index >= 0)
                    message << "layer " << layer_index << ' ';
                message << "has unsupported mixed projection arithmetic: ";
                for (std::size_t index = 0; index < specs.size(); ++index)
                {
                    const auto &spec = specs[index];
                    if (index != 0) message << ", ";
                    if (layer_index >= 0)
                        message << "blk." << layer_index << ".ffn_" << spec.label << "_exps.weight";
                    else
                        message << spec.label;
                    message << '=' << projectionFormatName(spec.format);
                }
                message << ". The movable payload requires all NativeQuantised projections "
                           "(different quantized codebooks are supported), or one shared "
                           "FP16/BF16/FP32 precision. Mixed quantized/floating or floating-precision "
                           "projections within a movable payload are not implemented; choose "
                           "a checkpoint with compatible projection families. No weights were converted.";
                throw std::invalid_argument(message.str());
            }
        }

        /** @return Exact bytes transmitted for this projection's active representation. */
        size_t projectionPayloadBytes(
            const DeviceMoETransferSlotDirectory::ProjectionSpec &spec)
        {
            if (spec.format.isFloating())
                return floatingBytes(spec, deviceFormat(spec.format));
            const size_t block_count =
                static_cast<size_t>(spec.N) * static_cast<size_t>(spec.K / 32);
            if (const auto coded = spec.codedDeclaration())
            {
                namespace sb = native_quantised_superblock;
                const auto bits = block_count * sb::superblockCodePlaneBits(coded->physical_codebook);
                return block_count * spec.payload_bytes_per_block + (bits + 7) / 8 +
                    sb::planeBytes(coded->physical_codebook, spec.N, spec.K / 32, coded->superblock_phase).superblock;
            }
            const auto planes = NativeQuantisedMetadataPlaneBytes::of(
                block_count, spec.scale_plane, spec.is_asymmetric, spec.has_emins);
            return block_count * static_cast<size_t>(spec.payload_bytes_per_block) +
                   planes.scales + planes.secondary + planes.emins;
        }

        /** @return Complete expert wire size for an exact layer tuple. */
        size_t expertPayloadBytes(
            const std::vector<DeviceMoETransferSlotDirectory::ProjectionSpec> &specs)
        {
            size_t bytes = 0;
            for (const auto &spec : specs)
            {
                const size_t projection_bytes = projectionPayloadBytes(spec);
                if (projection_bytes > std::numeric_limits<size_t>::max() - bytes)
                    throw std::overflow_error("Transfer directory expert payload overflows size_t");
                bytes += projection_bytes;
            }
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
                    if (spec.format.isFloating())
                    {
                        pool.planRawWeight(directorySlotName(slot, spec.label), spec.N, spec.K,
                                           floatingBytes(spec, deviceFormat(spec.format)));
                        continue;
                    }
                    pool.planWeight(
                        directorySlotName(slot, spec.label),
                        spec.N,
                        spec.K,
                        spec.payload_bytes_per_block,
                        spec.is_asymmetric,
                        spec.has_emins,
                        spec.scale_plane,
                        /*raw_gguf_bytes=*/0,
                        {}, spec.codedDeclaration());
                }
            }
        }
    } // namespace

    void validateGPUExpertProjectionContract(
        const std::vector<GpuExpertSlotPool::ProjectionSpec> &specs,
        DeviceMoEProjectionSet projection_set, int layer_index)
    {
        validateSpecs(specs, projection_set, layer_index);
    }

    DeviceMoETransferSlotDirectory::FormatProfile
    DeviceMoETransferSlotDirectory::profileForLayerFormats(
        const std::vector<std::vector<ProjectionSpec>> &layer_formats,
        DeviceMoEProjectionSet projection_set)
    {
        if (layer_formats.empty())
        {
            throw std::invalid_argument(
                "DeviceMoETransferSlotDirectory requires at least one layer format");
        }

        if (layer_formats.size() > kDeviceMoEMaxTransferSlots)
            throw std::invalid_argument("Transfer directory layer count exceeds its slot address space");

        FormatProfile profile{.projection_set = projection_set};
        for (size_t layer = 0; layer < layer_formats.size(); ++layer)
        {
            auto specs = layer_formats[layer];
            validateGPUExpertProjectionContract(specs, projection_set, static_cast<int>(layer));
            for (const auto &spec : specs)
            {
                const auto coded = spec.codedDeclaration();
                if (coded && coded->physical_codebook != spec.codebook_id)
                    throw std::invalid_argument("Transfer directory coded projection uses the wrong physical codebook");
            }
            std::sort(specs.begin(), specs.end(), [](const auto &a, const auto &b) {
                return a.label < b.label;
            });
            profile.max_wire_payload_bytes = std::max(profile.max_wire_payload_bytes, expertPayloadBytes(specs));
            const auto it = std::find_if(profile.families.begin(), profile.families.end(),
                [&](const SlotFamily &family) { return family.projections == specs; });
            const auto family = static_cast<uint32_t>(it - profile.families.begin());
            profile.layer_families.push_back(family);
            if (it == profile.families.end())
                profile.families.push_back({std::move(specs), 1u});
            else
                ++it->layer_count;
        }
        return profile;
    }

    DeviceMoETransferSlotDirectory::FormatProfile
    DeviceMoETransferSlotDirectory::profileForLayerWeightManifest(
        const std::vector<MoEOverlayLayerWeightManifest> &
            layer_weight_manifest,
        DeviceMoEProjectionSet projection_set)
    {
        if (layer_weight_manifest.empty())
        {
            throw std::invalid_argument(
                "DeviceMoETransferSlotDirectory requires a non-empty model manifest");
        }
        const int first_model_layer = layer_weight_manifest.front().layer_idx;
        if (first_model_layer < 0 ||
            layer_weight_manifest.size() > static_cast<std::size_t>(
                std::numeric_limits<int>::max() - first_model_layer))
            throw std::invalid_argument(
                "DeviceMoETransferSlotDirectory manifest has an invalid owned layer interval");

        std::vector<std::vector<ProjectionSpec>> layer_formats;
        layer_formats.reserve(layer_weight_manifest.size());
        for (std::size_t layer_offset = 0;
             layer_offset < layer_weight_manifest.size();
             ++layer_offset)
        {
            const auto &layer = layer_weight_manifest[layer_offset];
            if (!layer.valid() ||
                layer.layer_idx != first_model_layer + static_cast<int>(layer_offset))
            {
                throw std::invalid_argument(
                    "DeviceMoETransferSlotDirectory manifest must be valid and contiguous within its owned interval");
            }

            std::vector<ProjectionSpec> specs;
            specs.reserve(layer.projections.size());
            for (const auto &projection : layer.projections)
            {
                // The model manifest remains complete. Filter only by the
                // frozen ownership contract, never by a missing source pointer.
                if (projection_set == DeviceMoEProjectionSet::GateUp &&
                    projection.projection == ExpertTierWeightProjection::Down)
                    continue;
                if (projection.format.isFloating())
                {
                    specs.push_back({
                        .label = projectionLabel(projection.projection),
                        .N = projection.N,
                        .K = projection.K,
                        .format = projection.format,
                    });
                    continue;
                }
                const auto *source =
                    native_quantised_formats::forSourceIdentity(
                        projection.format.native_quantised.codebook_id,
                        projection.format.native_quantised.is_superblock);
                if (!source)
                {
                    throw std::invalid_argument(
                        "DeviceMoETransferSlotDirectory manifest contains an uncatalogued NativeQuantised source format");
                }

                // Quote the same migration layout used by the prepared engine.
                const auto reusable =
                    gpuPreparedWeightPoolAllocation(*source, GPUPreparedWeightPoolLayout::MigrationReusable);
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
                        canonicalDeviceVnniCodebookId(*source),
                    .scale_plane = nativeQuantisedScalePlaneWidth(
                        canonicalDeviceVnniCodebookId(*source)),
                    .format = projection.format,
                    .gpu_superblock_phase = projection.gpuSuperblockPhase(),
                });
            }
            layer_formats.push_back(std::move(specs));
        }
        return profileForLayerFormats(layer_formats, projection_set);
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
        auto capacity = planBufferedCapacity(
            std::max(
                persistentActiveSlotDemand(config),
                minimum_active_slots),
            transfer_wave_slots,
            transfer_buffer_count);
        capacity.active_slots_per_layer = std::max(1u, config.max_hot_replicas_per_participant);
        capacity.minimum_active_slots = static_cast<uint32_t>(minimum_active_slots);
        return capacity;
    }

    DeviceMoETransferSlotDirectory::AllocationBOM
    DeviceMoETransferSlotDirectory::allocationBOM(
        BufferedCapacity capacity,
        const FormatProfile &profile)
    {
        if (capacity.active_slots == 0u ||
            uint64_t{capacity.active_slots} + capacity.staging_slots != capacity.total_slots ||
            profile.families.empty() || profile.layer_families.empty())
            throw std::invalid_argument("Transfer directory quote requires coherent slot capacity and families");

        std::vector<uint32_t> members(profile.families.size());
        for (const auto family : profile.layer_families)
            ++members.at(family);

        AllocationBOM bom;
        for (size_t family_index = 0; family_index < profile.families.size(); ++family_index)
        {
            const auto &family = profile.families[family_index];
            if (family.layer_count == 0u || family.layer_count != members[family_index])
                throw std::invalid_argument("Transfer directory family membership disagrees with its layer mapping");
            const uint64_t active = capacity.active_slots_per_layer == 0u
                ? capacity.active_slots
                : std::max(uint64_t{capacity.active_slots_per_layer} * family.layer_count,
                           uint64_t{capacity.minimum_active_slots});
            const uint64_t count = active + capacity.staging_slots;
            if (count + bom.total_slots > kDeviceMoEMaxTransferSlots)
                throw std::invalid_argument("Transfer directory families exceed the device slot address space");
            const auto end = static_cast<uint32_t>(bom.total_slots + count);
            WeightVRAMPool planner;
            planDirectoryPayload(planner, static_cast<uint32_t>(count), family.projections);
            bom.families.push_back({
                .range = {bom.total_slots, end, static_cast<uint32_t>(active)},
                .payload_bytes = alignGPUWeightLoadAllocation(planner.totalPlannedBytes()),
                .descriptor_bytes = static_cast<size_t>(count) * sizeof(DeviceMoEExpertDirectoryEntry) * 2u,
            });
            const auto &quote = bom.families.back();
            if (quote.descriptor_bytes > std::numeric_limits<size_t>::max() - quote.payload_bytes ||
                quote.totalBytes() > std::numeric_limits<size_t>::max() - bom.total_bytes)
                throw std::overflow_error("Transfer directory family BOM overflows size_t");
            bom.total_slots = end;
            bom.active_slots += static_cast<uint32_t>(active);
            bom.payload_bytes += quote.payload_bytes;
            bom.descriptor_bytes += quote.descriptor_bytes;
            bom.total_bytes += quote.totalBytes();
        }
        bom.layer_range_bytes = profile.layer_families.size() * sizeof(DeviceMoETransferSlotRange);
        if (bom.layer_range_bytes > std::numeric_limits<size_t>::max() - bom.total_bytes)
            throw std::overflow_error("Transfer directory BOM overflows size_t");
        bom.total_bytes += bom.layer_range_bytes;
        return bom;
    }

    uint32_t DeviceMoETransferSlotDirectory::runtimeMinimumActiveSlots(bool native_gpu_transport,
        RoutedExpertAssignmentPolicy prefill_assignment, const PrefixCacheRuntimeConfig &prefix_cache,
        uint32_t compact_payload_slots) noexcept
    {
        return native_gpu_transport && (prefill_assignment == RoutedExpertAssignmentPolicy::LeastLoadedResident ||
            (prefix_cache.enabled && prefix_cache.moe_policy != PrefixCacheMoEPolicy::Disabled))
            ? compact_payload_slots : 0u;
    }

    std::string DeviceMoETransferSlotDirectory::captureIdentity(
        const FormatProfile &profile, BufferedCapacity capacity)
    {
        std::ostringstream key;
        key << ":transfer_histogram=" << int(moe_rebalance_abi::HistogramLayout::TransferFamilyOccupancy)
            << ":transfer=" << int(profile.projection_set) << ':' << capacity.active_slots
            << ':' << capacity.staging_slots << ':' << capacity.active_slots_per_layer
            << ':' << capacity.minimum_active_slots << ":wire=" << profile.max_wire_payload_bytes;
        for (const auto &family : profile.families)
        {
            key << ":family";
            for (const auto &spec : family.projections)
            {
                const auto coded = spec.codedDeclaration();
                const auto source = spec.format.native_quantised;
                key << ':' << spec.label << '=' << spec.N << 'x' << spec.K
                    << ',' << int(spec.format.kind) << ',' << source.present
                    << ',' << int(source.codebook_id) << ',' << source.is_superblock
                    << ',' << int(spec.codebook_id) << ',' << coded.has_value()
                    << ',' << int(spec.gpu_superblock_phase) << ',' << spec.payload_bytes_per_block
                    << ',' << int(spec.scale_plane) << ',' << spec.is_asymmetric << ',' << spec.has_emins;
            }
        }
        key << ":layers=";
        for (const auto family : profile.layer_families)
            key << family << ',';
        return key.str();
    }

    void DeviceMoETransferSlotDirectory::requireCaptureIdentity(
        const FormatProfile &profile, BufferedCapacity capacity) const
    {
        if (capture_identity_ != captureIdentity(profile, capacity))
            throw std::logic_error("Device rebalance requested a different transfer slot family layout for its retained directory");
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
        if (!explicit_test_allocation && (!memory_authority || capacity.staging_slots == 0u))
            throw std::invalid_argument("Transfer directory production allocation requires authority and staging capacity");

        auto bom = allocationBOM(capacity, format_profile);
        auto capture_identity = captureIdentity(format_profile, capacity);
        const uint32_t slot_count = bom.total_slots;
        std::vector<DeviceMoEExpertDirectoryEntry> host_entries(slot_count);
        std::vector<PhysicalMemoryAllocationLease> payload_leases;
        std::vector<std::unique_ptr<WeightVRAMPool>> pools;
        for (size_t family_index = 0; family_index < bom.families.size(); ++family_index)
        {
            const auto &family = bom.families[family_index];
            const auto &specs = format_profile.families[family_index].projections;
            const auto format = deviceFormat(specs.front().format);
            auto pool = std::make_unique<WeightVRAMPool>();
            planDirectoryPayload(*pool, family.range.end - family.range.first, specs);
            if (!explicit_test_allocation)
                payload_leases.push_back(memory_authority->claimNewAllocation(
                    device, PhysicalMemoryOwner::ExpertMigrationStaging, family.payload_bytes));
            if (!pool->allocate(backend, device_ordinal) || pool->totalPlannedBytes() != family.payload_bytes)
                throw std::runtime_error("Transfer directory family allocation disagrees with its quote");
            for (uint32_t slot = family.range.first; slot < family.range.end; ++slot)
            {
                DeviceMoEExpertDirectoryEntry entry;
                entry.layer = kDeviceMoEInvalidSlot;
                entry.expert = kDeviceMoEInvalidSlot;
                entry.participant = participant_id;
                entry.slot_index = slot;
                entry.descriptor.logical_expert_id = -1;
                // The directory participant owns physical storage. An F shard
                // never names that participant as a whole logical expert owner.
                entry.descriptor.owner_participant =
                    format_profile.projection_set == DeviceMoEProjectionSet::IntermediateShard
                        ? -1 : static_cast<int32_t>(participant_id);
                entry.descriptor.local_slot = static_cast<int32_t>(slot);
                entry.descriptor.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                                          DeviceMoEExpertFlags::TransferSlot);
                entry.descriptor.weight_format = format;
                entry.descriptor.floating_allocation_format = format;
                entry.descriptor.projection_set = format_profile.projection_set;
                entry.flags =
                    static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Valid) |
                    static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::TransferSlot);

                for (const auto &spec : specs)
                {
                    auto maybe_slot = pool->getSlot(directorySlotName(slot - family.range.first, spec.label));
                    if (!maybe_slot.has_value())
                        throw std::runtime_error("DeviceMoETransferSlotDirectory missing planned transfer slot");
                    setProjectionDescriptor(
                        entry.descriptor,
                        spec,
                        *maybe_slot,
                        format);
                }
                if (!deviceMoETransferSlotReadyForCopy(
                        entry,
                        participant_id,
                        /*layer=*/0,
                        /*expert=*/0))
                {
                    throw std::runtime_error(
                        "Transfer directory produced an unusable slot for family " + std::to_string(family_index) +
                        " (" + projectionFormatName(specs.front().format) + ")");
                }
                host_entries[slot] = entry;
            }
            pools.push_back(std::move(pool));
            logVramBomLine("moe_device_transfer_slot_family",
                "device=" + device.to_string() + " family=" + std::to_string(family_index) +
                " first_slot=" + std::to_string(family.range.first) +
                " end_slot=" + std::to_string(family.range.end) +
                " payload_bytes=" + std::to_string(family.payload_bytes) +
                " descriptor_bytes=" + std::to_string(family.descriptor_bytes));
        }

        std::vector<DeviceMoETransferSlotRange> host_ranges;
        for (const auto family : format_profile.layer_families)
            host_ranges.push_back(bom.families[family].range);
        const size_t directory_bytes = bom.descriptor_bytes / 2u;
        std::optional<PhysicalMemoryAllocationLease>
            directory_entries_lease;
        if (!explicit_test_allocation)
        {
            directory_entries_lease.emplace(
                memory_authority->claimNewAllocation(
                    device,
                    PhysicalMemoryOwner::ExpertMigrationStaging,
                    bom.descriptor_bytes + bom.layer_range_bytes));
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

        auto *device_layer_ranges = static_cast<DeviceMoETransferSlotRange *>(
            backend->allocate(bom.layer_range_bytes, device_ordinal));
        if (!device_layer_ranges)
        {
            backend->free(device_baseline_entries, device_ordinal);
            backend->free(device_entries, device_ordinal);
            throw std::runtime_error("Transfer directory failed to allocate layer ranges");
        }

        void *upload_stream = backend->createStream(device_ordinal);
        if (!upload_stream)
        {
            backend->free(device_layer_ranges, device_ordinal);
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
            backend->hostToDeviceOnStream(device_layer_ranges, host_ranges.data(),
                bom.layer_range_bytes, device_ordinal, upload_stream) &&
            backend->synchronizeStream(upload_stream, device_ordinal);
        backend->destroyStream(upload_stream, device_ordinal);
        if (!uploaded)
        {
            backend->free(device_layer_ranges, device_ordinal);
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
                std::move(bom),
                std::move(format_profile),
                std::move(capture_identity),
                std::move(payload_leases),
                std::move(pools),
                device_entries,
                device_baseline_entries,
                device_layer_ranges,
                std::move(host_entries),
                std::move(directory_entries_lease)));

        return directory;
    }

    DeviceMoETransferSlotDirectory::DeviceMoETransferSlotDirectory(
        IBackend *backend,
        DeviceId device,
        int device_ordinal,
        uint32_t participant_id,
        AllocationBOM bom,
        FormatProfile profile,
        std::string capture_identity,
        std::vector<PhysicalMemoryAllocationLease> payload_leases,
        std::vector<std::unique_ptr<WeightVRAMPool>> pools,
        DeviceMoEExpertDirectoryEntry *device_entries,
        DeviceMoEExpertDirectoryEntry *device_baseline_entries,
        DeviceMoETransferSlotRange *device_layer_ranges,
        std::vector<DeviceMoEExpertDirectoryEntry> host_entries,
        std::optional<PhysicalMemoryAllocationLease> directory_entries_lease)
        : backend_(backend),
          device_(device),
          device_ordinal_(device_ordinal),
          participant_id_(participant_id),
          slot_count_(bom.total_slots),
          active_slot_count_(bom.active_slots),
          bom_(std::move(bom)),
          profile_(std::move(profile)),
          capture_identity_(std::move(capture_identity)),
          payload_leases_(std::move(payload_leases)),
          pools_(std::move(pools)),
          device_entries_(device_entries),
          device_baseline_entries_(device_baseline_entries),
          device_layer_ranges_(device_layer_ranges),
          host_entries_(std::move(host_entries)),
          directory_entries_lease_(std::move(directory_entries_lease)),
          planned_bytes_(bom_.payload_bytes),
          wire_payload_bytes_(profile_.max_wire_payload_bytes)
    {
    }

    DeviceMoETransferSlotDirectory::~DeviceMoETransferSlotDirectory()
    {
        if (device_layer_ranges_)
            backend_->free(device_layer_ranges_, device_ordinal_);
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
        uint32_t layer,
        uint32_t logical_expert,
        DeviceMoEExpertDescriptor &out) const
    {
        if (layer >= profile_.layer_families.size())
            return false;
        const auto &range = bom_.families[profile_.layer_families[layer]].range;
        if (slot_index < range.first || slot_index >= range.end)
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

} // namespace llaminar2
