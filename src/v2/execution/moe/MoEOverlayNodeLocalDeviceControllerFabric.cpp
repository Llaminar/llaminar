/**
 * @file MoEOverlayNodeLocalDeviceControllerFabric.cpp
 * @brief POSIX mapping, NUMA placement, and GPU alias setup for controller RCU.
 *
 * This file executes before graph construction. It may wait for local MPI
 * peers, but it never observes live histogram data or participates in an
 * inference transaction. Once construction returns, CUDA/HIP kernels are the
 * only writers of controller lifecycle and policy records.
 */

#include "MoEOverlayNodeLocalDeviceControllerFabric.h"

#include "MoEOverlayEconomyProfileComposer.h"
#include "MoEOverlayServiceTelemetryPublication.h"

#include "../../utils/FNV1a.h"
#include "interfaces/IMPIContext.h"
#include "interfaces/IMPITopology.h"
#include "transfer/TransferEngine.h"
#include "utils/PerfStatsCollector.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>

namespace llaminar2
{
    namespace
    {
        constexpr std::uint64_t kSetupPublication = 1u;
        constexpr auto kSetupTimeout = std::chrono::seconds(30);

        /** @return `value` rounded upward to a power-of-two alignment. */
        std::size_t checkedAlignUp(
            std::size_t value,
            std::size_t alignment,
            const char *description)
        {
            if (alignment == 0u || (alignment & (alignment - 1u)) != 0u)
            {
                throw std::invalid_argument(
                    std::string("device controller fabric ") + description +
                    " alignment is not a power of two");
            }
            const std::size_t mask = alignment - 1u;
            if (value > std::numeric_limits<std::size_t>::max() - mask)
            {
                throw std::overflow_error(
                    std::string("device controller fabric ") + description +
                    " alignment overflows size_t");
            }
            return (value + mask) & ~mask;
        }

        /** @return Checked product used only for immutable mapping capacity. */
        std::size_t checkedMultiply(
            std::size_t left,
            std::size_t right,
            const char *description)
        {
            if (left != 0u &&
                right > std::numeric_limits<std::size_t>::max() / left)
            {
                throw std::overflow_error(
                    std::string("device controller fabric ") + description +
                    " overflows size_t");
            }
            return left * right;
        }

        /** @return Checked sum used only for immutable mapping capacity. */
        std::size_t checkedAdd(
            std::size_t left,
            std::size_t right,
            const char *description)
        {
            if (right > std::numeric_limits<std::size_t>::max() - left)
            {
                throw std::overflow_error(
                    std::string("device controller fabric ") + description +
                    " overflows size_t");
            }
            return left + right;
        }

        /** Acquire one shared setup scalar without changing its plain ABI. */
        template <typename T>
        T loadAcquire(T &value) noexcept
        {
            return std::atomic_ref<T>(value).load(std::memory_order_acquire);
        }

        /** Release one shared setup scalar without embedding std::atomic. */
        template <typename T>
        void storeRelease(T &target, T value) noexcept
        {
            std::atomic_ref<T>(target).store(value, std::memory_order_release);
        }

        /** @return Byte pointer advanced by a validated immutable offset. */
        template <typename T>
        T *at(std::byte *base, std::size_t offset) noexcept
        {
            return reinterpret_cast<T *>(base + offset);
        }

        /** @return Stable process-local device key for duplicate rejection. */
        std::string deviceKey(DeviceId device)
        {
            return device.toString();
        }

        /** @return Dense tier count represented by immutable participant metadata. */
        std::uint32_t denseTierCount(
            const MoEOverlayDeviceControllerTopology &topology)
        {
            std::vector<bool> observed;
            for (const auto &participant : topology.participants)
            {
                if (participant.tier_idx < 0)
                {
                    throw std::invalid_argument(
                        "device controller fabric requires non-negative dense tier indices");
                }
                const auto tier = static_cast<std::size_t>(
                    participant.tier_idx);
                if (tier >=
                    kMoEOverlayDeviceControllerFabricMaxParticipants)
                {
                    throw std::invalid_argument(
                        "device controller fabric tier index exceeds its fixed ABI");
                }
                if (observed.size() <= tier)
                    observed.resize(tier + 1u, false);
                observed[tier] = true;
            }
            if (observed.empty() ||
                std::find(observed.begin(), observed.end(), false) !=
                    observed.end())
            {
                throw std::invalid_argument(
                    "device controller fabric requires contiguous tier indices");
            }
            return static_cast<std::uint32_t>(observed.size());
        }

        /** @return Stable non-zero identity fingerprint for a certified profile. */
        std::uint64_t economyIdentityFingerprint(
            const std::string &identity) noexcept
        {
            const std::uint64_t hash = fnv1a64(
                identity.data(), identity.size(), kFNV1a64OffsetBasis);
            return hash == 0u ? 1u : hash;
        }

        /** @return Participant role bits consumed by device policy kernels. */
        std::uint32_t participantFlags(
            const MoEOverlayDeviceControllerTopology &topology,
            int participant_id,
            const MoEOverlayDeviceControllerGroup &group) noexcept
        {
            std::uint32_t flags = static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerParticipantFlags::None);
            if (participant_id == topology.leader_participant_id)
            {
                flags |= static_cast<std::uint32_t>(
                    MoEOverlayDeviceControllerParticipantFlags::AuthorityLeader);
            }
            if (participant_id == group.root_participant_id)
            {
                flags |= static_cast<std::uint32_t>(
                    MoEOverlayDeviceControllerParticipantFlags::GroupRoot);
            }
            if (group.group_id == topology.leader_group_id)
            {
                flags |= static_cast<std::uint32_t>(
                    MoEOverlayDeviceControllerParticipantFlags::
                        InferenceEpochMember);
            }
            return flags;
        }

        /** @return Run-scoped shared-memory name with no topology role words. */
        std::string makeChannelName(
            std::uint64_t node_namespace,
            std::uint64_t topology_fingerprint)
        {
            std::ostringstream name;
            name << "/llaminar_moectrl_" << std::hex << std::setw(16)
                 << std::setfill('0') << node_namespace << '_'
                 << std::setw(16) << topology_fingerprint;
            return name.str();
        }

        /**
         * @return Stable identity of the exact packed expert byte geometry.
         *
         * An empty setup vector means that Dynamic policy is intentionally not
         * armed yet; it hashes as an all-zero vector of the declared layer
         * count so every attaching rank still proves the same immutable ABI.
         */
        std::uint64_t payloadGeometryFingerprint(
            const std::vector<std::uint64_t> &bytes,
            std::uint32_t num_layers) noexcept
        {
            std::uint64_t hash = 1469598103934665603ULL;
            const auto mix = [&hash](std::uint64_t value)
            {
                for (std::uint32_t byte = 0u; byte < 8u; ++byte)
                {
                    hash ^= (value >> (byte * 8u)) & 0xffu;
                    hash *= 1099511628211ULL;
                }
            };
            mix(num_layers);
            for (std::uint32_t layer = 0u; layer < num_layers; ++layer)
                mix(bytes.empty() ? 0u : bytes[layer]);
            return hash == 0u ? 1u : hash;
        }
    } // namespace

    struct MoEOverlayNodeLocalDeviceControllerFabric::MappingLifetime
    {
        /** Unmap and close only after every GPU registration releases us. */
        ~MappingLifetime()
        {
            if (base && base != MAP_FAILED && bytes != 0u)
                (void)::munmap(base, bytes);
            if (fd >= 0)
                (void)::close(fd);
        }

        int fd = -1;
        void *base = nullptr;
        std::size_t bytes = 0u;
    };

    bool MoEOverlayDeviceControllerFabricLayout::valid() const noexcept
    {
        if (page_bytes == 0u ||
            (page_bytes & (page_bytes - 1u)) != 0u ||
            mapping_bytes == 0u || mapping_bytes % page_bytes != 0u ||
            setup_header_offset != 0u ||
            sizeof(MoEOverlayDeviceControllerFabricSetupHeader) > page_bytes ||
            leader_owned_begin != page_bytes ||
            leader_owned_begin >= leader_owned_end ||
            leader_owned_end > mapping_bytes ||
            leader_owned_begin % page_bytes != 0u ||
            leader_owned_end % page_bytes != 0u || groups.empty() ||
            header.magic != kMoEOverlayDeviceControllerFabricMagic ||
            header.version != kMoEOverlayDeviceControllerFabricVersion ||
            header.mapping_bytes != mapping_bytes ||
            header.group_count != groups.size() ||
            header.participant_count == 0u || header.num_layers == 0u ||
            header.num_experts == 0u || header.command_capacity == 0u ||
            header.routed_experts_per_token == 0u ||
            header.routed_experts_per_token > header.num_experts ||
            header.tier_count == 0u ||
            header.tier_count > header.participant_count ||
            header.payload_bytes_per_layer_offset < leader_owned_begin ||
            header.payload_bytes_per_layer_offset +
                    header.num_layers * sizeof(std::uint64_t) >
                leader_owned_end ||
            header.demand_history_offset < leader_owned_begin ||
            header.demand_history_offset <
                header.payload_bytes_per_layer_offset +
                    header.num_layers * sizeof(std::uint64_t) ||
            header.demand_history_offset % alignof(std::uint64_t) != 0u ||
            header.demand_history_words !=
                static_cast<std::uint64_t>(
                    kMoEOverlayDeviceControllerDemandPhaseCount) *
                    header.num_layers * header.num_experts ||
            header.demand_history_offset +
                    header.demand_history_words * sizeof(std::uint64_t) >
                leader_owned_end ||
            header.economy_header_offset <
                header.demand_history_offset +
                    header.demand_history_words * sizeof(std::uint64_t) ||
            header.economy_header_offset %
                    alignof(MoEOverlayDeviceControllerEconomyHeader) !=
                0u ||
            header.economy_header_offset +
                    sizeof(MoEOverlayDeviceControllerEconomyHeader) >
                leader_owned_end ||
            header.economy_service_cost_offset <
                header.economy_header_offset +
                    sizeof(MoEOverlayDeviceControllerEconomyHeader) ||
            header.economy_service_cost_words !=
                static_cast<std::uint64_t>(header.tier_count) *
                    header.num_layers *
                    kMoEOverlayDeviceControllerEconomyServicePhaseCount ||
            header.economy_service_cost_offset +
                    header.economy_service_cost_words *
                        sizeof(std::uint64_t) >
                leader_owned_end ||
            header.economy_migration_cost_offset <
                header.economy_service_cost_offset +
                    header.economy_service_cost_words *
                        sizeof(std::uint64_t) ||
            header.economy_migration_cost_offset %
                    alignof(MoEOverlayDeviceControllerMigrationCost) !=
                0u ||
            header.economy_migration_cost_entries !=
                static_cast<std::uint64_t>(header.participant_count) *
                    header.participant_count * header.num_layers ||
            header.economy_migration_cost_offset +
                    header.economy_migration_cost_entries *
                        sizeof(MoEOverlayDeviceControllerMigrationCost) >
                leader_owned_end ||
            header.economy_last_moved_offset <
                header.economy_migration_cost_offset +
                    header.economy_migration_cost_entries *
                        sizeof(MoEOverlayDeviceControllerMigrationCost) ||
            header.economy_last_moved_offset % alignof(std::uint64_t) != 0u ||
            header.economy_last_moved_words !=
                static_cast<std::uint64_t>(header.num_layers) *
                    header.num_experts ||
            header.economy_last_moved_offset +
                    header.economy_last_moved_words *
                        sizeof(std::uint64_t) >
                leader_owned_end ||
            header.inference_epoch_record_offset < leader_owned_begin ||
            header.inference_epoch_record_offset %
                    alignof(
                        MoEOverlayDeviceControllerInferenceEpochRecord) !=
                0u ||
            header.inference_epoch_record_offset <
                header.controller_header_offset +
                    sizeof(MoEOverlayDeviceControllerSharedHeader) ||
            header.inference_epoch_record_offset +
                    sizeof(MoEOverlayDeviceControllerInferenceEpochRecord) >
                leader_owned_end ||
            header.command_header_offset <
                header.inference_epoch_record_offset +
                    sizeof(
                        MoEOverlayDeviceControllerInferenceEpochRecord) ||
            header.minimum_window_activations == 0u ||
            header.maximum_cycles_per_wave == 0u ||
            header.payload_geometry_fingerprint == 0u)
        {
            return false;
        }

        std::size_t previous_end = leader_owned_end;
        for (std::size_t index = 0; index < groups.size(); ++index)
        {
            const auto &group = groups[index];
            if (group.group_id != index || group.participant_count == 0u ||
                group.participant_count >
                    kMoEOverlayDeviceControllerFabricMaxParticipants ||
                group.root_world_rank < 0 ||
                group.owned_page_begin != previous_end ||
                group.owned_page_begin % page_bytes != 0u ||
                group.owned_page_end % page_bytes != 0u ||
                group.owned_page_begin >= group.owned_page_end ||
                group.owned_page_end > mapping_bytes ||
                group.participant_record_count != group.participant_count ||
                group.participant_records_offset < group.owned_page_begin ||
                group.participant_records_offset +
                        group.participant_record_count *
                            sizeof(MoEOverlayDeviceControllerParticipantRecord) >
                    group.owned_page_end ||
                group.group_record_offset < group.owned_page_begin ||
                group.group_record_offset +
                        sizeof(MoEOverlayDeviceControllerGroupRecord) >
                    group.owned_page_end ||
                group.transport_record_offset < group.owned_page_begin ||
                group.transport_record_offset +
                        sizeof(MoEOverlayDeviceControllerTransportRecord) >
                    group.owned_page_end ||
                group.collected_state_offset < group.owned_page_begin ||
                group.collected_state_words == 0u ||
                group.collected_state_offset +
                        group.collected_state_words * sizeof(std::uint64_t) >
                    group.owned_page_end ||
                group.service_telemetry_offset < group.owned_page_begin ||
                group.service_telemetry_stride_bytes <
                    deviceMoEOverlayServiceTelemetryPublicationBytes(
                        header.num_layers) ||
                group.service_telemetry_bytes !=
                    group.service_telemetry_stride_bytes *
                        group.participant_count ||
                group.service_telemetry_offset +
                        group.service_telemetry_bytes >
                    group.owned_page_end)
            {
                return false;
            }
            previous_end = static_cast<std::size_t>(group.owned_page_end);
        }
        return previous_end == mapping_bytes;
    }

    bool MoEOverlayDeviceControllerParticipantBinding::valid() const noexcept
    {
        return device.is_gpu() && participant_id >= 0 && group_id >= 0 &&
               mapped_base_device && mapped_bytes != 0u && layout &&
               participants && groups && controller &&
               inference_epoch_record && command &&
               command_entries && payload_bytes_per_layer && demand_history &&
               economy && economy_service_costs &&
               economy_migration_costs && economy_last_moved &&
               local_group &&
               local_transport &&
               group_participant_records && local_participant_record &&
               group_collected_state &&
               participant_collected_state &&
               service_telemetry_publication && lifetime;
    }

    MoEOverlayDeviceControllerDeviceBinding
    MoEOverlayDeviceControllerParticipantBinding::deviceBinding() const
    {
        if (!valid())
        {
            throw std::logic_error(
                "device controller participant cannot export an incomplete GPU binding");
        }
        const auto &metadata = participants[participant_id];
        return {
            .mapped_base = mapped_base_device,
            .mapped_bytes = mapped_bytes,
            .layout = layout,
            .participants = participants,
            .groups = groups,
            .controller = controller,
            .inference_epoch_record = inference_epoch_record,
            .command = command,
            .command_entries = command_entries,
            .payload_bytes_per_layer = payload_bytes_per_layer,
            .demand_history = demand_history,
            .economy = economy,
            .economy_service_costs = economy_service_costs,
            .economy_migration_costs = economy_migration_costs,
            .economy_last_moved = economy_last_moved,
            .local_group = local_group,
            .local_transport = local_transport,
            .group_participant_records = group_participant_records,
            .local_participant_record = local_participant_record,
            .group_collected_state = group_collected_state,
            .participant_collected_state = participant_collected_state,
            .topology_fingerprint = layout->topology_fingerprint,
            .participant_id = static_cast<std::uint32_t>(participant_id),
            .group_id = static_cast<std::uint32_t>(group_id),
            .role_flags = metadata.flags,
        };
    }

    MoEOverlayDeviceControllerFabricLayout
    MoEOverlayNodeLocalDeviceControllerFabric::planLayout(
        const MoEOverlayDeviceControllerTopology &topology,
        std::uint32_t num_layers,
        std::uint32_t num_experts,
        std::uint32_t command_capacity,
        std::size_t page_bytes,
        const std::vector<std::uint64_t> &payload_bytes_per_layer,
        std::uint64_t minimum_window_activations,
        std::uint32_t maximum_cycles_per_wave,
        std::uint32_t dynamic_imbalance_threshold_per_mille,
        std::uint32_t dynamic_minimum_improvement_per_mille,
        std::uint32_t dynamic_maximum_cycles_per_layer,
        std::uint32_t dynamic_maximum_commands_per_wave,
        std::uint32_t routed_experts_per_token)
    {
        if (!topology.valid() || num_layers == 0u || num_experts == 0u ||
            num_layers > kMoEOverlayDeviceControllerFabricMaxLayers ||
            num_experts > kMoEOverlayDeviceControllerFabricMaxExperts ||
            command_capacity == 0u || routed_experts_per_token == 0u ||
            routed_experts_per_token > num_experts ||
            topology.participants.size() >
                kMoEOverlayDeviceControllerFabricMaxParticipants ||
            topology.groups.size() >
                kMoEOverlayDeviceControllerFabricMaxParticipants ||
            (!payload_bytes_per_layer.empty() &&
             payload_bytes_per_layer.size() != num_layers) ||
            minimum_window_activations == 0u ||
            maximum_cycles_per_wave == 0u ||
            page_bytes <
                sizeof(MoEOverlayDeviceControllerFabricSetupHeader))
        {
            throw std::invalid_argument(
                "device controller fabric layout requires supported positive topology and geometry");
        }

        MoEOverlayDeviceControllerFabricLayout result;
        const std::uint32_t tier_count = denseTierCount(topology);
        result.page_bytes = page_bytes;
        result.setup_header_offset = 0u;
        result.leader_owned_begin = page_bytes;
        std::size_t cursor = result.leader_owned_begin;

        const std::size_t layout_header_offset = cursor;
        cursor = checkedAdd(
            cursor,
            sizeof(MoEOverlayDeviceControllerFabricLayoutHeader),
            "layout header");
        cursor = checkedAlignUp(cursor, 64u, "participant metadata");
        const std::size_t participant_metadata_offset = cursor;
        cursor = checkedAdd(
            cursor,
            checkedMultiply(
                topology.participants.size(),
                sizeof(MoEOverlayDeviceControllerParticipantMetadata),
                "participant metadata bytes"),
            "participant metadata");
        cursor = checkedAlignUp(cursor, 64u, "group layout");
        const std::size_t group_layout_offset = cursor;
        cursor = checkedAdd(
            cursor,
            checkedMultiply(
                topology.groups.size(),
                sizeof(MoEOverlayDeviceControllerFabricGroupLayout),
                "group layout bytes"),
            "group layout");
        cursor = checkedAlignUp(cursor, 64u, "controller header");
        const std::size_t controller_header_offset = cursor;
        cursor = checkedAdd(
            cursor,
            sizeof(MoEOverlayDeviceControllerSharedHeader),
            "controller header");
        cursor = checkedAlignUp(cursor, 64u, "inference epoch record");
        const std::size_t inference_epoch_record_offset = cursor;
        cursor = checkedAdd(
            cursor,
            sizeof(MoEOverlayDeviceControllerInferenceEpochRecord),
            "inference epoch record");
        cursor = checkedAlignUp(cursor, 64u, "command header");
        const std::size_t command_header_offset = cursor;
        cursor = checkedAdd(
            cursor,
            sizeof(MoEOverlayDeviceControllerCommandHeader),
            "command header");
        cursor = checkedAlignUp(cursor, 64u, "command entries");
        const std::size_t command_entries_offset = cursor;
        cursor = checkedAdd(
            cursor,
            checkedMultiply(
                command_capacity,
                sizeof(MoEOverlayDeviceMovementCommand),
                "command entry bytes"),
            "command entries");
        cursor = checkedAlignUp(cursor, 64u, "payload byte geometry");
        const std::size_t payload_bytes_per_layer_offset = cursor;
        cursor = checkedAdd(
            cursor,
            checkedMultiply(
                num_layers,
                sizeof(std::uint64_t),
                "payload byte geometry bytes"),
            "payload byte geometry");
        cursor = checkedAlignUp(cursor, 64u, "phase demand history");
        const std::size_t demand_history_offset = cursor;
        const std::size_t demand_history_words = checkedMultiply(
            kMoEOverlayDeviceControllerDemandPhaseCount,
            checkedMultiply(
                num_layers,
                num_experts,
                "phase demand layer/expert words"),
            "phase demand history words");
        cursor = checkedAdd(
            cursor,
            checkedMultiply(
                demand_history_words,
                sizeof(std::uint64_t),
                "phase demand history bytes"),
            "phase demand history");
        cursor = checkedAlignUp(cursor, 64u, "economy header");
        const std::size_t economy_header_offset = cursor;
        cursor = checkedAdd(
            cursor,
            sizeof(MoEOverlayDeviceControllerEconomyHeader),
            "economy header");
        cursor = checkedAlignUp(cursor, 64u, "economy service costs");
        const std::size_t economy_service_cost_offset = cursor;
        const std::size_t economy_service_cost_words = checkedMultiply(
            tier_count,
            checkedMultiply(
                num_layers,
                kMoEOverlayDeviceControllerEconomyServicePhaseCount,
                "economy layer/phase service words"),
            "economy tier service words");
        cursor = checkedAdd(
            cursor,
            checkedMultiply(
                economy_service_cost_words,
                sizeof(std::uint64_t),
                "economy service bytes"),
            "economy service costs");
        cursor = checkedAlignUp(
            cursor,
            alignof(MoEOverlayDeviceControllerMigrationCost),
            "economy migration costs");
        const std::size_t economy_migration_cost_offset = cursor;
        const std::size_t economy_migration_cost_entries = checkedMultiply(
            topology.participants.size(),
            checkedMultiply(
                topology.participants.size(),
                num_layers,
                "economy destination/layer movement entries"),
            "economy source movement entries");
        cursor = checkedAdd(
            cursor,
            checkedMultiply(
                economy_migration_cost_entries,
                sizeof(MoEOverlayDeviceControllerMigrationCost),
                "economy migration bytes"),
            "economy migration costs");
        cursor = checkedAlignUp(cursor, 64u, "economy movement history");
        const std::size_t economy_last_moved_offset = cursor;
        const std::size_t economy_last_moved_words = checkedMultiply(
            num_layers,
            num_experts,
            "economy movement history words");
        cursor = checkedAdd(
            cursor,
            checkedMultiply(
                economy_last_moved_words,
                sizeof(std::uint64_t),
                "economy movement history bytes"),
            "economy movement history");
        result.leader_owned_end =
            checkedAlignUp(cursor, page_bytes, "leader page range");
        cursor = result.leader_owned_end;

        result.groups.reserve(topology.groups.size());
        for (const auto &topology_group : topology.groups)
        {
            if (!topology_group.valid() ||
                topology_group.participant_ids.size() >
                    kMoEOverlayDeviceControllerFabricMaxParticipants)
            {
                throw std::invalid_argument(
                    "device controller fabric group exceeds the shared runtime ABI");
            }
            MoEOverlayDeviceControllerFabricGroupLayout group;
            group.group_id = static_cast<std::uint32_t>(
                topology_group.group_id);
            group.participant_count = static_cast<std::uint32_t>(
                topology_group.participant_ids.size());
            group.root_participant_id = static_cast<std::uint32_t>(
                topology_group.root_participant_id);
            group.root_world_rank = topology_group.root_world_rank;
            group.tier_index = topology_group.tier_index;
            group.tier_priority = topology_group.tier_priority;
            group.intra_group_transport = static_cast<std::uint32_t>(
                topology_group.intra_group_transport);
            group.inter_group_transport = static_cast<std::uint32_t>(
                topology_group.inter_group_transport);
            for (std::size_t index = 0;
                 index < topology_group.participant_ids.size(); ++index)
            {
                group.participant_ids[index] = static_cast<std::uint32_t>(
                    topology_group.participant_ids[index]);
            }

            group.owned_page_begin = cursor;
            group.participant_records_offset = cursor;
            group.participant_record_count = group.participant_count;
            cursor = checkedAdd(
                cursor,
                checkedMultiply(
                    group.participant_count,
                    sizeof(MoEOverlayDeviceControllerParticipantRecord),
                    "participant record bytes"),
                "participant records");
            cursor = checkedAlignUp(cursor, 64u, "group record");
            group.group_record_offset = cursor;
            cursor = checkedAdd(
                cursor,
                sizeof(MoEOverlayDeviceControllerGroupRecord),
                "group record");
            cursor = checkedAlignUp(cursor, 64u, "transport record");
            group.transport_record_offset = cursor;
            cursor = checkedAdd(
                cursor,
                sizeof(MoEOverlayDeviceControllerTransportRecord),
                "transport record");
            cursor = checkedAlignUp(cursor, 64u, "collected state");
            group.collected_state_offset = cursor;
            const std::size_t state_words = checkedMultiply(
                topology_group.participant_ids.size(),
                checkedMultiply(
                    num_layers,
                    num_experts,
                    "layer/expert state words"),
                "group member state words");
            group.collected_state_words = state_words;
            cursor = checkedAdd(
                cursor,
                checkedMultiply(
                    state_words,
                    sizeof(std::uint64_t),
                    "collected state bytes"),
                "collected state");
            cursor = checkedAlignUp(
                cursor, 64u, "service telemetry publications");
            group.service_telemetry_offset = cursor;
            group.service_telemetry_stride_bytes = checkedAlignUp(
                deviceMoEOverlayServiceTelemetryPublicationBytes(
                    num_layers),
                64u,
                "service telemetry participant stride");
            group.service_telemetry_bytes = checkedMultiply(
                group.participant_count,
                group.service_telemetry_stride_bytes,
                "service telemetry group bytes");
            cursor = checkedAdd(
                cursor,
                static_cast<std::size_t>(group.service_telemetry_bytes),
                "service telemetry publications");
            cursor = checkedAlignUp(cursor, page_bytes, "group page range");
            group.owned_page_end = cursor;
            result.groups.push_back(group);
        }

        result.mapping_bytes = cursor;
        result.header = {
            .participant_count = static_cast<std::uint32_t>(
                topology.participants.size()),
            .group_count = static_cast<std::uint32_t>(topology.groups.size()),
            .num_layers = num_layers,
            .num_experts = num_experts,
            .command_capacity = command_capacity,
            .tier_count = tier_count,
            .topology_fingerprint = topology.topology_fingerprint,
            .mapping_bytes = result.mapping_bytes,
            .participant_metadata_offset = participant_metadata_offset,
            .group_layout_offset = group_layout_offset,
            .controller_header_offset = controller_header_offset,
            .inference_epoch_record_offset = inference_epoch_record_offset,
            .command_header_offset = command_header_offset,
            .command_entries_offset = command_entries_offset,
            .payload_bytes_per_layer_offset =
                payload_bytes_per_layer_offset,
            .demand_history_offset = demand_history_offset,
            .demand_history_words = demand_history_words,
            .economy_header_offset = economy_header_offset,
            .economy_service_cost_offset = economy_service_cost_offset,
            .economy_service_cost_words = economy_service_cost_words,
            .economy_migration_cost_offset =
                economy_migration_cost_offset,
            .economy_migration_cost_entries =
                economy_migration_cost_entries,
            .economy_last_moved_offset = economy_last_moved_offset,
            .economy_last_moved_words = economy_last_moved_words,
            .minimum_window_activations = minimum_window_activations,
            .maximum_cycles_per_wave = maximum_cycles_per_wave,
            .dynamic_imbalance_threshold_per_mille =
                dynamic_imbalance_threshold_per_mille,
            .dynamic_minimum_improvement_per_mille =
                dynamic_minimum_improvement_per_mille,
            .dynamic_maximum_cycles_per_layer =
                dynamic_maximum_cycles_per_layer,
            .dynamic_maximum_commands_per_wave =
                dynamic_maximum_commands_per_wave,
            .payload_geometry_fingerprint = payloadGeometryFingerprint(
                payload_bytes_per_layer, num_layers),
            .routed_experts_per_token = routed_experts_per_token,
        };
        if (layout_header_offset != result.leader_owned_begin || !result.valid())
        {
            throw std::logic_error(
                "device controller fabric produced an inconsistent page layout");
        }
        return result;
    }

    MoEOverlayNodeLocalDeviceControllerFabric::
        MoEOverlayNodeLocalDeviceControllerFabric(Config config)
        : config_(std::move(config))
    {
        if (!config_.mpi_ctx || !config_.topology ||
            !config_.topology->valid() || config_.num_layers == 0u ||
            config_.num_experts == 0u || config_.command_capacity == 0u ||
            config_.routed_experts_per_token == 0u ||
            config_.routed_experts_per_token > config_.num_experts ||
            config_.initial_durable_epoch == 0u ||
            (!config_.payload_bytes_per_layer.empty() &&
             config_.payload_bytes_per_layer.size() != config_.num_layers) ||
            config_.minimum_window_activations == 0u ||
            config_.maximum_cycles_per_wave == 0u)
        {
            throw std::invalid_argument(
                "node-local device controller fabric requires complete topology and geometry");
        }
        const auto *const mpi_topology = config_.mpi_ctx->topology();
        if (!mpi_topology || mpi_topology->node_shared_memory_namespace() == 0u)
        {
            throw std::invalid_argument(
                "node-local device controller fabric requires a run-scoped physical-node namespace");
        }

        layout_ = planLayout(
            *config_.topology,
            config_.num_layers,
            config_.num_experts,
            config_.command_capacity,
            4096u,
            config_.payload_bytes_per_layer,
            config_.minimum_window_activations,
            config_.maximum_cycles_per_wave,
            config_.dynamic_imbalance_threshold_per_mille,
            config_.dynamic_minimum_improvement_per_mille,
            config_.dynamic_maximum_cycles_per_layer,
            config_.dynamic_maximum_commands_per_wave,
            config_.routed_experts_per_token);

        for (const auto &group : config_.topology->groups)
            participating_world_ranks_.push_back(group.root_world_rank);
        std::sort(
            participating_world_ranks_.begin(),
            participating_world_ranks_.end());
        participating_world_ranks_.erase(
            std::unique(
                participating_world_ranks_.begin(),
                participating_world_ranks_.end()),
            participating_world_ranks_.end());
        if (participating_world_ranks_.empty() ||
            participating_world_ranks_.size() >
                kMoEOverlayDeviceControllerFabricMaxParticipants ||
            !std::binary_search(
                participating_world_ranks_.begin(),
                participating_world_ranks_.end(),
                config_.topology->leader_world_rank))
        {
            throw std::invalid_argument(
                "node-local device controller fabric has invalid participating-rank identity");
        }

        std::unordered_set<std::string> seen_devices;
        const int local_rank = config_.mpi_ctx->rank();
        for (const auto &participant : config_.topology->participants)
        {
            if (participant.world_rank != local_rank)
                continue;
            local_participant_ids_.push_back(participant.participant_id);
            if (seen_devices.insert(deviceKey(participant.device)).second)
                local_devices_.push_back(participant.device);
        }
        if (local_participant_ids_.empty() || local_devices_.empty() ||
            !std::binary_search(
                participating_world_ranks_.begin(),
                participating_world_ranks_.end(),
                local_rank))
        {
            throw std::invalid_argument(
                "node-local device controller fabric may be built only by a routed participant rank");
        }
        for (const DeviceId device : local_devices_)
        {
            if (!device.is_gpu())
            {
                throw std::invalid_argument(
                    "node-local device controller fabric cannot register a non-GPU endpoint");
            }
        }

        channel_name_ = makeChannelName(
            mpi_topology->node_shared_memory_namespace(),
            config_.topology->topology_fingerprint);
        try
        {
            mapOrAttach();
            placeOwnedPages();
            initializeControllerRecords();
            registerLocalDevices();
        }
        catch (...)
        {
            publishSetupError(
                MoEOverlayDeviceControllerFabricSetupError::MappingFailure);
            mapped_region_.reset();
            mapping_lifetime_.reset();
            if (creator_)
                (void)::shm_unlink(channel_name_.c_str());
            throw;
        }
    }

    MoEOverlayNodeLocalDeviceControllerFabric::
        ~MoEOverlayNodeLocalDeviceControllerFabric()
    {
        // POSIX unlink only removes the name; every already-open mapping stays
        // valid until its process-local driver registrations are released.
        if (creator_ && !channel_name_.empty())
            (void)::shm_unlink(channel_name_.c_str());
        mapped_region_.reset();
        mapping_lifetime_.reset();
    }

    void MoEOverlayNodeLocalDeviceControllerFabric::mapOrAttach()
    {
        const int local_rank = config_.mpi_ctx->rank();
        creator_ = local_rank == config_.topology->leader_world_rank;
        int fd = -1;
        if (creator_)
        {
            fd = ::shm_open(
                channel_name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
            if (fd < 0)
            {
                throw std::runtime_error(
                    "device controller fabric creator could not create " +
                    channel_name_ + ": " + std::strerror(errno));
            }
            if (::ftruncate(fd, static_cast<off_t>(layout_.mapping_bytes)) != 0)
            {
                const std::string detail = std::strerror(errno);
                (void)::close(fd);
                (void)::shm_unlink(channel_name_.c_str());
                throw std::runtime_error(
                    "device controller fabric ftruncate failed: " + detail);
            }
        }
        else
        {
            const auto deadline = std::chrono::steady_clock::now() +
                                  kSetupTimeout;
            do
            {
                fd = ::shm_open(channel_name_.c_str(), O_RDWR, 0);
                if (fd >= 0)
                    break;
                if (errno != ENOENT)
                {
                    throw std::runtime_error(
                        "device controller fabric follower could not open " +
                        channel_name_ + ": " + std::strerror(errno));
                }
                std::this_thread::yield();
            } while (std::chrono::steady_clock::now() < deadline);
            if (fd < 0)
            {
                throw std::runtime_error(
                    "device controller fabric timed out waiting for its leader mapping");
            }
        }

        struct stat status{};
        const auto size_deadline = std::chrono::steady_clock::now() +
                                   kSetupTimeout;
        for (;;)
        {
            if (::fstat(fd, &status) != 0)
            {
                const std::string detail = std::strerror(errno);
                (void)::close(fd);
                throw std::runtime_error(
                    "device controller fabric fstat failed: " + detail);
            }
            if (status.st_size >= 0 &&
                static_cast<std::size_t>(status.st_size) >=
                    layout_.mapping_bytes)
            {
                break;
            }
            if (std::chrono::steady_clock::now() >= size_deadline)
            {
                (void)::close(fd);
                throw std::runtime_error(
                    "device controller fabric timed out waiting for mapping size");
            }
            std::this_thread::yield();
        }
        if (status.st_size < 0 ||
            static_cast<std::size_t>(status.st_size) != layout_.mapping_bytes)
        {
            (void)::close(fd);
            throw std::runtime_error(
                "device controller fabric mapping size disagrees across ranks");
        }

        void *const base = ::mmap(
            nullptr,
            layout_.mapping_bytes,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            fd,
            0);
        if (base == MAP_FAILED)
        {
            const std::string detail = std::strerror(errno);
            (void)::close(fd);
            throw std::runtime_error(
                "device controller fabric mmap failed: " + detail);
        }
        mapping_lifetime_ = std::make_shared<MappingLifetime>();
        mapping_lifetime_->fd = fd;
        mapping_lifetime_->base = base;
        mapping_lifetime_->bytes = layout_.mapping_bytes;

#if defined(MADV_HUGEPAGE)
        // Group regions remain page isolated even when the kernel chooses a
        // huge-page backing; first-touch still determines their NUMA home.
        (void)::madvise(base, layout_.mapping_bytes, MADV_HUGEPAGE);
#endif

        auto *const setup = static_cast<
            MoEOverlayDeviceControllerFabricSetupHeader *>(base);
        if (creator_)
        {
            std::memset(base, 0, layout_.page_bytes);
            *setup = MoEOverlayDeviceControllerFabricSetupHeader{};
            setup->topology_fingerprint =
                config_.topology->topology_fingerprint;
            setup->node_namespace =
                config_.mpi_ctx->topology()->node_shared_memory_namespace();
            setup->mapping_bytes = layout_.mapping_bytes;
            setup->num_layers = config_.num_layers;
            setup->num_experts = config_.num_experts;
            setup->routed_experts_per_token =
                config_.routed_experts_per_token;
            setup->participant_count = static_cast<std::uint32_t>(
                config_.topology->participants.size());
            setup->group_count = static_cast<std::uint32_t>(
                config_.topology->groups.size());
            setup->command_capacity = config_.command_capacity;
            setup->participating_rank_count = static_cast<std::uint32_t>(
                participating_world_ranks_.size());
            setup->leader_world_rank =
                config_.topology->leader_world_rank;
            for (std::size_t index = 0;
                 index < participating_world_ranks_.size(); ++index)
            {
                setup->participating_world_ranks[index] =
                    participating_world_ranks_[index];
            }
            storeRelease(
                setup->state,
                static_cast<std::uint32_t>(
                    MoEOverlayDeviceControllerFabricSetupState::LayoutPublished));
        }
        else
        {
            const auto deadline = std::chrono::steady_clock::now() +
                                  kSetupTimeout;
            while (loadAcquire(setup->state) == static_cast<std::uint32_t>(
                       MoEOverlayDeviceControllerFabricSetupState::Uninitialized))
            {
                if (std::chrono::steady_clock::now() >= deadline)
                {
                    throw std::runtime_error(
                        "device controller fabric timed out waiting for layout publication");
                }
                std::this_thread::yield();
            }
        }

        if (setup->magic != kMoEOverlayDeviceControllerFabricMagic ||
            setup->version != kMoEOverlayDeviceControllerFabricVersion ||
            setup->topology_fingerprint !=
                config_.topology->topology_fingerprint ||
            setup->node_namespace !=
                config_.mpi_ctx->topology()->node_shared_memory_namespace() ||
            setup->mapping_bytes != layout_.mapping_bytes ||
            setup->num_layers != config_.num_layers ||
            setup->num_experts != config_.num_experts ||
            setup->routed_experts_per_token !=
                config_.routed_experts_per_token ||
            setup->participant_count != config_.topology->participants.size() ||
            setup->group_count != config_.topology->groups.size() ||
            setup->command_capacity != config_.command_capacity ||
            setup->participating_rank_count !=
                participating_world_ranks_.size() ||
            setup->leader_world_rank !=
                config_.topology->leader_world_rank ||
            !std::equal(
                participating_world_ranks_.begin(),
                participating_world_ranks_.end(),
                setup->participating_world_ranks))
        {
            publishSetupError(
                MoEOverlayDeviceControllerFabricSetupError::TopologyMismatch);
            throw std::runtime_error(
                "device controller fabric immutable setup identity disagrees across ranks");
        }
    }

    void MoEOverlayNodeLocalDeviceControllerFabric::placeOwnedPages()
    {
        auto *const base = static_cast<std::byte *>(mapping_lifetime_->base);
        auto *const setup = reinterpret_cast<
            MoEOverlayDeviceControllerFabricSetupHeader *>(base);
        const int local_rank = config_.mpi_ctx->rank();

        std::size_t touched_bytes = 0u;
        if (local_rank == config_.topology->leader_world_rank)
        {
            std::memset(
                base + layout_.leader_owned_begin,
                0,
                layout_.leader_owned_end - layout_.leader_owned_begin);
            touched_bytes +=
                layout_.leader_owned_end - layout_.leader_owned_begin;
        }
        for (const auto &group : layout_.groups)
        {
            if (group.root_world_rank != local_rank)
                continue;
            const auto begin = static_cast<std::size_t>(
                group.owned_page_begin);
            const auto end = static_cast<std::size_t>(group.owned_page_end);
            std::memset(base + begin, 0, end - begin);
            touched_bytes += end - begin;

            auto *const record = at<MoEOverlayDeviceControllerGroupRecord>(
                base,
                static_cast<std::size_t>(group.group_record_offset));
            *record = MoEOverlayDeviceControllerGroupRecord{};
            record->group_id = group.group_id;
            record->root_participant_id = group.root_participant_id;
            record->topology_fingerprint =
                config_.topology->topology_fingerprint;

            // Physical transport has one disjoint writer lane. It never shares
            // the GPU-owned group acknowledgement cache lines above.
            auto *const transport = at<
                MoEOverlayDeviceControllerTransportRecord>(
                base,
                static_cast<std::size_t>(
                    group.transport_record_offset));
            *transport = MoEOverlayDeviceControllerTransportRecord{};
            transport->group_id = group.group_id;
            transport->topology_fingerprint =
                config_.topology->topology_fingerprint;

            auto *const participant_records = at<
                MoEOverlayDeviceControllerParticipantRecord>(
                base,
                static_cast<std::size_t>(
                    group.participant_records_offset));
            for (std::uint32_t member = 0u;
                 member < group.participant_count;
                 ++member)
            {
                participant_records[member] =
                    MoEOverlayDeviceControllerParticipantRecord{};
                participant_records[member].participant_id =
                    group.participant_ids[member];
                participant_records[member].group_id = group.group_id;
                participant_records[member].topology_fingerprint =
                    config_.topology->topology_fingerprint;

                auto *const service = at<
                    MoEOverlayDeviceServiceTelemetryPublicationHeader>(
                    base,
                    checkedAdd(
                        static_cast<std::size_t>(
                            group.service_telemetry_offset),
                        checkedMultiply(
                            member,
                            static_cast<std::size_t>(
                                group.service_telemetry_stride_bytes),
                            "service telemetry member offset"),
                        "service telemetry member address"));
                *service =
                    MoEOverlayDeviceServiceTelemetryPublicationHeader{};
                service->participant_id = static_cast<std::int32_t>(
                    group.participant_ids[member]);
                service->layer_count = config_.num_layers;
            }
        }

        const auto rank_position = std::lower_bound(
            participating_world_ranks_.begin(),
            participating_world_ranks_.end(),
            local_rank);
        if (rank_position == participating_world_ranks_.end() ||
            *rank_position != local_rank)
        {
            throw std::logic_error(
                "device controller fabric local rank is absent from its rendezvous");
        }
        const auto index = static_cast<std::size_t>(
            rank_position - participating_world_ranks_.begin());
        storeRelease(setup->first_touch_ready[index], kSetupPublication);
        waitForRankPublications(
            setup->first_touch_ready,
            kSetupPublication,
            "NUMA first-touch");
        if (creator_)
        {
            storeRelease(
                setup->state,
                static_cast<std::uint32_t>(
                    MoEOverlayDeviceControllerFabricSetupState::PagesPlaced));
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "device_fabric_first_touch_bytes",
            static_cast<double>(touched_bytes),
            "model_setup",
            "node_local",
            {{"world_rank", std::to_string(local_rank)},
             {"topology_fingerprint",
              std::to_string(config_.topology->topology_fingerprint)}});
    }

    void MoEOverlayNodeLocalDeviceControllerFabric::
        initializeControllerRecords()
    {
        auto *const base = static_cast<std::byte *>(mapping_lifetime_->base);
        auto *const setup = reinterpret_cast<
            MoEOverlayDeviceControllerFabricSetupHeader *>(base);
        auto *const controller = at<MoEOverlayDeviceControllerSharedHeader>(
            base,
            static_cast<std::size_t>(layout_.header.controller_header_offset));

        if (creator_)
        {
            *at<MoEOverlayDeviceControllerFabricLayoutHeader>(
                base, layout_.leader_owned_begin) = layout_.header;
            auto *const participant_metadata = at<
                MoEOverlayDeviceControllerParticipantMetadata>(
                base,
                static_cast<std::size_t>(
                    layout_.header.participant_metadata_offset));
            auto *const group_layout = at<
                MoEOverlayDeviceControllerFabricGroupLayout>(
                base,
                static_cast<std::size_t>(
                    layout_.header.group_layout_offset));
            std::copy(
                layout_.groups.begin(),
                layout_.groups.end(),
                group_layout);

            for (const auto &participant : config_.topology->participants)
            {
                const auto *const group =
                    config_.topology->groupForParticipant(
                        participant.participant_id);
                if (!group)
                {
                    publishSetupError(
                        MoEOverlayDeviceControllerFabricSetupError::InvalidLayout);
                    throw std::logic_error(
                        "device controller fabric cannot bind participant metadata to a group");
                }
                participant_metadata[participant.participant_id] = {
                    .participant_id = static_cast<std::uint32_t>(
                        participant.participant_id),
                    .group_id = static_cast<std::uint32_t>(group->group_id),
                    .tier_index = participant.tier_idx,
                    .tier_priority = group->tier_priority,
                    .world_rank = participant.world_rank,
                    .device_type = static_cast<std::uint32_t>(
                        participant.device.type),
                    .device_ordinal = participant.device.ordinal,
                    .domain_participant_index = static_cast<std::uint32_t>(
                        participant.domain_participant_index),
                    .flags = participantFlags(
                        *config_.topology,
                        participant.participant_id,
                        *group),
                };
            }

            *controller = MoEOverlayDeviceControllerSharedHeader{};
            controller->group_count = static_cast<std::uint32_t>(
                config_.topology->groups.size());
            controller->leader_group_id = static_cast<std::uint32_t>(
                config_.topology->leader_group_id);
            controller->leader_participant_id = static_cast<std::uint32_t>(
                config_.topology->leader_participant_id);
            controller->topology_fingerprint =
                config_.topology->topology_fingerprint;
            controller->current_durable_epoch =
                config_.initial_durable_epoch;
            controller->admission_epoch = config_.initial_durable_epoch;

            const auto &inference_group = config_.topology->groups.at(
                static_cast<std::size_t>(
                    config_.topology->leader_group_id));
            auto *const inference_epoch = at<
                MoEOverlayDeviceControllerInferenceEpochRecord>(
                base,
                static_cast<std::size_t>(
                    layout_.header.inference_epoch_record_offset));
            *inference_epoch =
                MoEOverlayDeviceControllerInferenceEpochRecord{};
            inference_epoch->publisher_participant_id =
                static_cast<std::uint32_t>(
                    config_.topology->leader_participant_id);
            inference_epoch->topology_fingerprint =
                config_.topology->topology_fingerprint;
            inference_epoch->epoch = config_.initial_durable_epoch;
            for (const int participant_id :
                 inference_group.participant_ids)
            {
                if (participant_id < 0 ||
                    participant_id >= static_cast<int>(
                        kMoEOverlayDeviceControllerInferenceEpochMaxParticipants))
                {
                    throw std::logic_error(
                        "device controller continuation epoch member exceeds its fixed ABI");
                }
                inference_epoch->participant_mask |=
                    1u << static_cast<std::uint32_t>(participant_id);
            }

            auto *const command = at<
                MoEOverlayDeviceControllerCommandHeader>(
                base,
                static_cast<std::size_t>(
                    layout_.header.command_header_offset));
            *command = MoEOverlayDeviceControllerCommandHeader{};
            command->topology_fingerprint =
                config_.topology->topology_fingerprint;

            auto *const payload_bytes = at<std::uint64_t>(
                base,
                static_cast<std::size_t>(
                    layout_.header.payload_bytes_per_layer_offset));
            for (std::uint32_t layer = 0u;
                 layer < config_.num_layers;
                 ++layer)
            {
                payload_bytes[layer] =
                    config_.payload_bytes_per_layer.empty()
                        ? 0u
                        : config_.payload_bytes_per_layer[layer];
            }
            auto *const demand_history = at<std::uint64_t>(
                base,
                static_cast<std::size_t>(
                    layout_.header.demand_history_offset));
            std::fill_n(
                demand_history,
                static_cast<std::size_t>(
                    layout_.header.demand_history_words),
                std::uint64_t{0u});

            auto *const economy = at<
                MoEOverlayDeviceControllerEconomyHeader>(
                base,
                static_cast<std::size_t>(
                    layout_.header.economy_header_offset));
            *economy = MoEOverlayDeviceControllerEconomyHeader{};
            economy->tier_count = layout_.header.tier_count;
            economy->participant_count =
                layout_.header.participant_count;
            economy->layer_count = layout_.header.num_layers;
            economy->topology_fingerprint =
                config_.topology->topology_fingerprint;
            auto *const economy_service = at<std::uint64_t>(
                base,
                static_cast<std::size_t>(
                    layout_.header.economy_service_cost_offset));
            std::fill_n(
                economy_service,
                static_cast<std::size_t>(
                    layout_.header.economy_service_cost_words),
                std::uint64_t{0u});
            auto *const economy_migration = at<
                MoEOverlayDeviceControllerMigrationCost>(
                base,
                static_cast<std::size_t>(
                    layout_.header.economy_migration_cost_offset));
            std::fill_n(
                economy_migration,
                static_cast<std::size_t>(
                    layout_.header.economy_migration_cost_entries),
                MoEOverlayDeviceControllerMigrationCost{});
            auto *const economy_last_moved = at<std::uint64_t>(
                base,
                static_cast<std::size_t>(
                    layout_.header.economy_last_moved_offset));
            std::fill_n(
                economy_last_moved,
                static_cast<std::size_t>(
                    layout_.header.economy_last_moved_words),
                kMoEOverlayDeviceControllerNeverMovedGeneration);

            // Idle is the release edge authenticating every immutable record,
            // including group records initialized on their owning ranks.
            storeRelease(
                controller->state,
                static_cast<std::uint32_t>(
                    MoEOverlayDeviceControllerState::Idle));
            storeRelease(
                setup->state,
                static_cast<std::uint32_t>(
                    MoEOverlayDeviceControllerFabricSetupState::
                        ControllerInitialized));
        }
        else
        {
            const auto deadline = std::chrono::steady_clock::now() +
                                  kSetupTimeout;
            while (loadAcquire(setup->state) < static_cast<std::uint32_t>(
                       MoEOverlayDeviceControllerFabricSetupState::
                           ControllerInitialized))
            {
                if (loadAcquire(setup->state) == static_cast<std::uint32_t>(
                        MoEOverlayDeviceControllerFabricSetupState::Error) ||
                    std::chrono::steady_clock::now() >= deadline)
                {
                    throw std::runtime_error(
                        "device controller fabric timed out waiting for controller initialization");
                }
                std::this_thread::yield();
            }
        }

        const auto *const published_layout = at<
            MoEOverlayDeviceControllerFabricLayoutHeader>(
            base, layout_.leader_owned_begin);
        const auto *const published_inference_epoch = at<
            MoEOverlayDeviceControllerInferenceEpochRecord>(
            base,
            static_cast<std::size_t>(
                layout_.header.inference_epoch_record_offset));
        std::uint32_t expected_inference_mask = 0u;
        for (const int participant_id :
             config_.topology->groups.at(
                 static_cast<std::size_t>(
                     config_.topology->leader_group_id))
                 .participant_ids)
        {
            expected_inference_mask |=
                1u << static_cast<std::uint32_t>(participant_id);
        }
        if (published_layout->magic !=
                kMoEOverlayDeviceControllerFabricMagic ||
            published_layout->version !=
                kMoEOverlayDeviceControllerFabricVersion ||
            published_layout->inference_epoch_record_offset !=
                layout_.header.inference_epoch_record_offset ||
            published_layout->payload_bytes_per_layer_offset !=
                layout_.header.payload_bytes_per_layer_offset ||
            published_layout->minimum_window_activations !=
                layout_.header.minimum_window_activations ||
            published_layout->maximum_cycles_per_wave !=
                layout_.header.maximum_cycles_per_wave ||
            published_layout->dynamic_imbalance_threshold_per_mille !=
                layout_.header.dynamic_imbalance_threshold_per_mille ||
            published_layout->dynamic_minimum_improvement_per_mille !=
                layout_.header.dynamic_minimum_improvement_per_mille ||
            published_layout->dynamic_maximum_cycles_per_layer !=
                layout_.header.dynamic_maximum_cycles_per_layer ||
            published_layout->dynamic_maximum_commands_per_wave !=
                layout_.header.dynamic_maximum_commands_per_wave ||
            published_layout->payload_geometry_fingerprint !=
                layout_.header.payload_geometry_fingerprint ||
            controller->magic != kMoEOverlayDeviceControllerMagic ||
            controller->version != kMoEOverlayDeviceControllerVersion ||
            controller->group_count != config_.topology->groups.size() ||
            controller->leader_group_id !=
                static_cast<std::uint32_t>(
                    config_.topology->leader_group_id) ||
            controller->leader_participant_id !=
                static_cast<std::uint32_t>(
                    config_.topology->leader_participant_id) ||
            controller->topology_fingerprint !=
                config_.topology->topology_fingerprint ||
            published_inference_epoch->magic !=
                kMoEOverlayDeviceControllerInferenceEpochMagic ||
            published_inference_epoch->version !=
                kMoEOverlayDeviceControllerInferenceEpochVersion ||
            published_inference_epoch->participant_mask !=
                expected_inference_mask ||
            published_inference_epoch->publisher_participant_id !=
                static_cast<std::uint32_t>(
                    config_.topology->leader_participant_id) ||
            published_inference_epoch->topology_fingerprint !=
                config_.topology->topology_fingerprint ||
            loadAcquire(controller->state) != static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerState::Idle))
        {
            publishSetupError(
                MoEOverlayDeviceControllerFabricSetupError::TopologyMismatch);
            throw std::runtime_error(
                "device controller fabric controller identity is incomplete");
        }
    }

    void MoEOverlayNodeLocalDeviceControllerFabric::registerLocalDevices()
    {
        auto *const base = static_cast<std::byte *>(mapping_lifetime_->base);
        auto *const setup = reinterpret_cast<
            MoEOverlayDeviceControllerFabricSetupHeader *>(base);
        try
        {
            TransferEngine transfer_engine;
            mapped_region_ = transfer_engine.registerExternalMappedHostRegion(
                base,
                layout_.mapping_bytes,
                local_devices_,
                mapping_lifetime_);
            if (!mapped_region_ || !mapped_region_->isBound())
            {
                throw std::runtime_error(
                    "device controller fabric did not bind every local GPU alias");
            }
        }
        catch (...)
        {
            publishSetupError(
                MoEOverlayDeviceControllerFabricSetupError::RegistrationFailure);
            throw;
        }

        const int local_rank = config_.mpi_ctx->rank();
        const auto rank_position = std::lower_bound(
            participating_world_ranks_.begin(),
            participating_world_ranks_.end(),
            local_rank);
        const auto index = static_cast<std::size_t>(
            rank_position - participating_world_ranks_.begin());
        storeRelease(setup->registration_ready[index], kSetupPublication);
        waitForRankPublications(
            setup->registration_ready,
            kSetupPublication,
            "GPU page registration");
        if (creator_)
        {
            storeRelease(
                setup->state,
                static_cast<std::uint32_t>(
                    MoEOverlayDeviceControllerFabricSetupState::Registered));
        }
        else
        {
            const auto deadline = std::chrono::steady_clock::now() +
                                  kSetupTimeout;
            while (loadAcquire(setup->state) != static_cast<std::uint32_t>(
                       MoEOverlayDeviceControllerFabricSetupState::Registered))
            {
                if (loadAcquire(setup->state) == static_cast<std::uint32_t>(
                        MoEOverlayDeviceControllerFabricSetupState::Error) ||
                    std::chrono::steady_clock::now() >= deadline)
                {
                    throw std::runtime_error(
                        "device controller fabric timed out waiting for global registration admission");
                }
                std::this_thread::yield();
            }
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "device_fabric_registered",
            1.0,
            "model_setup",
            "node_local",
            {{"world_rank", std::to_string(config_.mpi_ctx->rank())},
             {"local_devices", std::to_string(local_devices_.size())},
             {"groups", std::to_string(layout_.groups.size())},
             {"mapping_bytes", std::to_string(layout_.mapping_bytes)},
             {"host_policy", "false"}});
    }

    void MoEOverlayNodeLocalDeviceControllerFabric::waitForRankPublications(
        const std::uint64_t *publications,
        std::uint64_t expected,
        const char *description) const
    {
        if (!publications || expected == 0u || !mapping_lifetime_)
        {
            throw std::invalid_argument(
                "device controller fabric publication wait is incomplete");
        }
        auto *const setup = static_cast<
            MoEOverlayDeviceControllerFabricSetupHeader *>(
            mapping_lifetime_->base);
        const auto deadline = std::chrono::steady_clock::now() +
                              kSetupTimeout;
        for (;;)
        {
            bool ready = true;
            for (std::size_t index = 0;
                 index < participating_world_ranks_.size(); ++index)
            {
                ready = ready &&
                        loadAcquire(
                            const_cast<std::uint64_t &>(publications[index])) ==
                            expected;
            }
            if (ready)
                return;
            if (loadAcquire(setup->state) == static_cast<std::uint32_t>(
                    MoEOverlayDeviceControllerFabricSetupState::Error))
            {
                throw std::runtime_error(
                    std::string("device controller fabric peer failed during ") +
                    description);
            }
            if (std::chrono::steady_clock::now() >= deadline)
            {
                throw std::runtime_error(
                    std::string("device controller fabric timed out during ") +
                    description);
            }
            std::this_thread::yield();
        }
    }

    void MoEOverlayNodeLocalDeviceControllerFabric::publishSetupError(
        MoEOverlayDeviceControllerFabricSetupError error) noexcept
    {
        if (!mapping_lifetime_ || !mapping_lifetime_->base ||
            error == MoEOverlayDeviceControllerFabricSetupError::None)
        {
            return;
        }
        auto *const setup = static_cast<
            MoEOverlayDeviceControllerFabricSetupHeader *>(
            mapping_lifetime_->base);
        std::uint32_t expected = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerFabricSetupError::None);
        (void)std::atomic_ref<std::uint32_t>(setup->error_code)
            .compare_exchange_strong(
                expected,
                static_cast<std::uint32_t>(error),
                std::memory_order_acq_rel,
                std::memory_order_acquire);
        storeRelease(
            setup->state,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerFabricSetupState::Error));
    }

    MoEOverlayDeviceControllerParticipantBinding
    MoEOverlayNodeLocalDeviceControllerFabric::participantBinding(
        int participant_id) const
    {
        if (!mapped_region_ || !mapped_region_->isBound() ||
            participant_id < 0 ||
            static_cast<std::size_t>(participant_id) >=
                config_.topology->participants.size())
        {
            throw std::out_of_range(
                "device controller fabric participant binding is unavailable");
        }
        const auto &participant = config_.topology->participants[
            static_cast<std::size_t>(participant_id)];
        if (participant.participant_id != participant_id ||
            participant.world_rank != config_.mpi_ctx->rank() ||
            !mapped_region_->hasDevice(participant.device))
        {
            throw std::out_of_range(
                "device controller fabric participant belongs to another rank or alias");
        }
        const auto *const topology_group =
            config_.topology->groupForParticipant(participant_id);
        if (!topology_group || topology_group->group_id < 0 ||
            static_cast<std::size_t>(topology_group->group_id) >=
                layout_.groups.size())
        {
            throw std::logic_error(
                "device controller fabric participant has no immutable group");
        }
        const auto &group = layout_.groups[
            static_cast<std::size_t>(topology_group->group_id)];
        const auto member = std::find(
            topology_group->participant_ids.begin(),
            topology_group->participant_ids.end(),
            participant_id);
        if (member == topology_group->participant_ids.end())
        {
            throw std::logic_error(
                "device controller fabric group membership diverged from topology");
        }
        const std::size_t member_index = static_cast<std::size_t>(
            member - topology_group->participant_ids.begin());
        const std::size_t words_per_participant = checkedMultiply(
            config_.num_layers,
            config_.num_experts,
            "participant collected-state words");

        auto *const base = static_cast<std::byte *>(
            mapped_region_->deviceAlias(participant.device));
        MoEOverlayDeviceControllerParticipantBinding binding{
            .device = participant.device,
            .participant_id = participant_id,
            .group_id = topology_group->group_id,
            .authority_leader =
                participant_id == config_.topology->leader_participant_id,
            .group_root =
                participant_id == topology_group->root_participant_id,
            .inference_epoch_member =
                topology_group->group_id ==
                config_.topology->leader_group_id,
            .mapped_base_device = base,
            .mapped_bytes = layout_.mapping_bytes,
            .layout = at<MoEOverlayDeviceControllerFabricLayoutHeader>(
                base, layout_.leader_owned_begin),
            .participants = at<
                MoEOverlayDeviceControllerParticipantMetadata>(
                base,
                static_cast<std::size_t>(
                    layout_.header.participant_metadata_offset)),
            .groups = at<MoEOverlayDeviceControllerFabricGroupLayout>(
                base,
                static_cast<std::size_t>(
                    layout_.header.group_layout_offset)),
            .controller = at<MoEOverlayDeviceControllerSharedHeader>(
                base,
                static_cast<std::size_t>(
                    layout_.header.controller_header_offset)),
            .inference_epoch_record = at<
                MoEOverlayDeviceControllerInferenceEpochRecord>(
                base,
                static_cast<std::size_t>(
                    layout_.header.inference_epoch_record_offset)),
            .command = at<MoEOverlayDeviceControllerCommandHeader>(
                base,
                static_cast<std::size_t>(
                    layout_.header.command_header_offset)),
            .command_entries = at<MoEOverlayDeviceMovementCommand>(
                base,
                static_cast<std::size_t>(
                    layout_.header.command_entries_offset)),
            .payload_bytes_per_layer = at<std::uint64_t>(
                base,
                static_cast<std::size_t>(
                    layout_.header.payload_bytes_per_layer_offset)),
            .demand_history = at<std::uint64_t>(
                base,
                static_cast<std::size_t>(
                    layout_.header.demand_history_offset)),
            .economy = at<MoEOverlayDeviceControllerEconomyHeader>(
                base,
                static_cast<std::size_t>(
                    layout_.header.economy_header_offset)),
            .economy_service_costs = at<std::uint64_t>(
                base,
                static_cast<std::size_t>(
                    layout_.header.economy_service_cost_offset)),
            .economy_migration_costs = at<
                MoEOverlayDeviceControllerMigrationCost>(
                base,
                static_cast<std::size_t>(
                    layout_.header.economy_migration_cost_offset)),
            .economy_last_moved = at<std::uint64_t>(
                base,
                static_cast<std::size_t>(
                    layout_.header.economy_last_moved_offset)),
            .local_group = at<MoEOverlayDeviceControllerGroupRecord>(
                base,
                static_cast<std::size_t>(group.group_record_offset)),
            .local_transport = at<
                MoEOverlayDeviceControllerTransportRecord>(
                base,
                static_cast<std::size_t>(
                    group.transport_record_offset)),
            .group_participant_records = at<
                MoEOverlayDeviceControllerParticipantRecord>(
                base,
                static_cast<std::size_t>(
                    group.participant_records_offset)),
            .local_participant_record = at<
                MoEOverlayDeviceControllerParticipantRecord>(
                base,
                checkedAdd(
                    static_cast<std::size_t>(
                        group.participant_records_offset),
                    checkedMultiply(
                        member_index,
                        sizeof(MoEOverlayDeviceControllerParticipantRecord),
                        "participant record offset"),
                    "participant record address")),
            .group_collected_state = at<std::uint64_t>(
                base,
                static_cast<std::size_t>(group.collected_state_offset)),
            .participant_collected_state = at<std::uint64_t>(
                base,
                checkedAdd(
                    static_cast<std::size_t>(group.collected_state_offset),
                    checkedMultiply(
                        member_index,
                        checkedMultiply(
                            words_per_participant,
                            sizeof(std::uint64_t),
                            "participant collected-state bytes"),
                        "participant collected-state offset"),
                    "participant collected-state address")),
            .service_telemetry_publication = at<
                MoEOverlayDeviceServiceTelemetryPublicationHeader>(
                base,
                checkedAdd(
                    static_cast<std::size_t>(
                        group.service_telemetry_offset),
                    checkedMultiply(
                        member_index,
                        static_cast<std::size_t>(
                            group.service_telemetry_stride_bytes),
                        "participant service telemetry bytes"),
                    "participant service telemetry address")),
            .lifetime = std::static_pointer_cast<const void>(mapped_region_),
        };
        if (!binding.valid())
        {
            throw std::logic_error(
                "device controller fabric produced an incomplete participant binding");
        }
        return binding;
    }

    std::string MoEOverlayNodeLocalDeviceControllerFabric::
        describeInferenceEpochBarrier() const
    {
        std::ostringstream description;
        if (!mapping_lifetime_ || !mapping_lifetime_->base ||
            mapping_lifetime_->base == MAP_FAILED || !layout_.valid())
        {
            return "controller_epoch_barrier{unavailable}";
        }

        auto *const base = static_cast<std::byte *>(
            mapping_lifetime_->base);
        auto *const controller = at<MoEOverlayDeviceControllerSharedHeader>(
            base,
            static_cast<std::size_t>(
                layout_.header.controller_header_offset));
        auto *const record = at<
            MoEOverlayDeviceControllerInferenceEpochRecord>(
            base,
            static_cast<std::size_t>(
                layout_.header.inference_epoch_record_offset));
        auto *const command = at<MoEOverlayDeviceControllerCommandHeader>(
            base,
            static_cast<std::size_t>(
                layout_.header.command_header_offset));

        const std::uint32_t mask = loadAcquire(record->participant_mask);
        description
            << "controller_epoch_barrier{controller_state="
            << loadAcquire(controller->state)
            << ",durable="
            << loadAcquire(controller->current_durable_epoch)
            << ",admission=" << loadAcquire(controller->admission_epoch)
            << ",transaction=" << loadAcquire(controller->transaction_id)
            << ",magic=" << loadAcquire(record->magic)
            << ",version=" << loadAcquire(record->version)
            << ",mask=0x" << std::hex << mask << std::dec
            << ",publisher="
            << loadAcquire(record->publisher_participant_id)
            << ",frozen_epoch=" << loadAcquire(record->epoch)
            << ",published_sequence="
            << loadAcquire(record->publication_sequence)
            << ",arrivals=[";
        bool first = true;
        for (std::uint32_t participant = 0u;
             participant <
                 kMoEOverlayDeviceControllerInferenceEpochMaxParticipants;
             ++participant)
        {
            if ((mask & (1u << participant)) == 0u)
                continue;
            if (!first)
                description << ',';
            first = false;
            description << participant << ':'
                        << loadAcquire(
                               record->arrival_sequence[participant]);
        }
        description
            << "],command{transaction="
            << loadAcquire(command->transaction_id)
            << ",count=" << loadAcquire(command->command_count)
            << ",demand_phase=" << loadAcquire(command->demand_phase)
            << ",parallel="
            << loadAcquire(command->parallel_command_count)
            << ",rounds=" << loadAcquire(command->movement_round_count)
            << ",cycles=" << loadAcquire(command->accepted_cycles)
            << ",promotions=" << loadAcquire(command->promotions)
            << ",demotions=" << loadAcquire(command->demotions)
            << ",same_priority="
            << loadAcquire(command->same_priority_moves)
            << ",bytes=" << loadAcquire(command->packed_weight_bytes)
            << "},transports=[";
        for (std::size_t group_index = 0u;
             group_index < layout_.groups.size();
             ++group_index)
        {
            if (group_index != 0u)
                description << ',';
            const auto &group = layout_.groups[group_index];
            auto *const transport = at<
                MoEOverlayDeviceControllerTransportRecord>(
                base,
                static_cast<std::size_t>(
                    group.transport_record_offset));
            description
                << group.group_id << ":state="
                << loadAcquire(transport->state)
                << ",status=" << loadAcquire(transport->status_code)
                << ",command="
                << loadAcquire(transport->command_transaction)
                << ",prepared="
                << loadAcquire(transport->prepared_transaction)
                << ",published="
                << loadAcquire(transport->published_transaction)
                << ",retired=" << loadAcquire(transport->retired_epoch);
        }
        description << "]}";
        return description.str();
    }

    std::uint64_t MoEOverlayNodeLocalDeviceControllerFabric::
        completedDurableMovementEpochs() const noexcept
    {
        if (!mapping_lifetime_ || !mapping_lifetime_->base ||
            mapping_lifetime_->base == MAP_FAILED || !layout_.valid())
        {
            return 0u;
        }

        auto *const controller = at<MoEOverlayDeviceControllerSharedHeader>(
            static_cast<std::byte *>(mapping_lifetime_->base),
            static_cast<std::size_t>(
                layout_.header.controller_header_offset));
        const std::uint64_t durable_epoch = loadAcquire(
            controller->current_durable_epoch);
        return durable_epoch >= config_.initial_durable_epoch
                   ? durable_epoch - config_.initial_durable_epoch
                   : 0u;
    }

    bool MoEOverlayNodeLocalDeviceControllerFabric::
        ownsEconomyPublication() const noexcept
    {
        return config_.mpi_ctx && config_.topology &&
               config_.mpi_ctx->rank() ==
                   config_.topology->leader_world_rank;
    }

    void MoEOverlayNodeLocalDeviceControllerFabric::
        publishCertifiedEconomyProfiles(
            const MoEOverlayCertifiedEconomyProfiles &profiles)
    {
        if (!ownsEconomyPublication())
        {
            throw std::logic_error(
                "only the frozen device-authority rank may publish measured economy evidence");
        }
        if (!profiles.valid() || !mapping_lifetime_ ||
            !mapping_lifetime_->base ||
            mapping_lifetime_->base == MAP_FAILED || !layout_.valid())
        {
            throw std::invalid_argument(
                "device controller economy publication requires complete certified profiles and a live fabric");
        }

        auto *const base = static_cast<std::byte *>(
            mapping_lifetime_->base);
        auto *const economy = at<
            MoEOverlayDeviceControllerEconomyHeader>(
            base,
            static_cast<std::size_t>(
                layout_.header.economy_header_offset));
        if (loadAcquire(economy->state) != static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerEconomyState::Empty))
        {
            throw std::logic_error(
                "device controller economy evidence may be published exactly once");
        }
        if (economy->magic != kMoEOverlayDeviceControllerFabricMagic ||
            economy->version !=
                kMoEOverlayDeviceControllerFabricVersion ||
            economy->tier_count != layout_.header.tier_count ||
            economy->participant_count !=
                layout_.header.participant_count ||
            economy->layer_count != layout_.header.num_layers ||
            economy->service_phase_count !=
                kMoEOverlayDeviceControllerEconomyServicePhaseCount ||
            economy->topology_fingerprint !=
                layout_.header.topology_fingerprint)
        {
            throw std::logic_error(
                "device controller economy header diverged from frozen topology geometry");
        }

        auto *const service = at<std::uint64_t>(
            base,
            static_cast<std::size_t>(
                layout_.header.economy_service_cost_offset));
        std::vector<bool> service_seen(
            static_cast<std::size_t>(
                layout_.header.economy_service_cost_words),
            false);
        if (profiles.service->costs.size() !=
            static_cast<std::size_t>(layout_.header.tier_count) *
                layout_.header.num_layers)
        {
            throw std::invalid_argument(
                "certified device economy service profile has incomplete tier/layer geometry");
        }
        std::uint32_t active_source_bits = 0u;
        for (std::uint32_t phase = 0u;
             phase <
                 kMoEOverlayDeviceControllerEconomyServicePhaseCount;
             ++phase)
        {
            if (profiles.service->active_sources[phase])
                active_source_bits |= 1u << phase;
        }
        if (active_source_bits == 0u)
        {
            throw std::invalid_argument(
                "certified device economy service profile enables no production phase");
        }
        for (const auto &row : profiles.service->costs)
        {
            if (row.tier_index < 0 || row.layer < 0 ||
                static_cast<std::uint32_t>(row.tier_index) >=
                    layout_.header.tier_count ||
                static_cast<std::uint32_t>(row.layer) >=
                    layout_.header.num_layers)
            {
                throw std::invalid_argument(
                    "certified device economy service profile contains an invalid coordinate");
            }
            for (std::uint32_t phase = 0u;
                 phase <
                     kMoEOverlayDeviceControllerEconomyServicePhaseCount;
                 ++phase)
            {
                const std::size_t offset =
                    (static_cast<std::size_t>(row.tier_index) *
                         layout_.header.num_layers +
                     static_cast<std::size_t>(row.layer)) *
                        kMoEOverlayDeviceControllerEconomyServicePhaseCount +
                    phase;
                const bool active = (active_source_bits & (1u << phase)) != 0u;
                const std::uint64_t cost =
                    row.nanoseconds_per_activation[phase];
                if (service_seen[offset] ||
                    (active && cost == 0u) || (!active && cost != 0u))
                {
                    throw std::invalid_argument(
                        "certified device economy service profile is duplicate or phase-inconsistent");
                }
                service[offset] = cost;
                service_seen[offset] = true;
            }
        }
        if (std::find(service_seen.begin(), service_seen.end(), false) !=
            service_seen.end())
        {
            throw std::invalid_argument(
                "certified device economy service profile omitted a coordinate");
        }

        auto *const migration = at<
            MoEOverlayDeviceControllerMigrationCost>(
            base,
            static_cast<std::size_t>(
                layout_.header.economy_migration_cost_offset));
        const std::size_t participant_count =
            layout_.header.participant_count;
        const std::size_t layer_count = layout_.header.num_layers;
        std::vector<bool> migration_seen(
            static_cast<std::size_t>(
                layout_.header.economy_migration_cost_entries),
            false);
        for (std::size_t participant = 0u;
             participant < participant_count;
             ++participant)
        {
            for (std::size_t layer = 0u; layer < layer_count; ++layer)
            {
                migration_seen[
                    (participant * participant_count + participant) *
                        layer_count +
                    layer] = true;
            }
        }
        if (profiles.migration->costs.size() !=
            participant_count * (participant_count - 1u) * layer_count)
        {
            throw std::invalid_argument(
                "certified device economy migration profile has incomplete directed geometry");
        }
        for (const auto &row : profiles.migration->costs)
        {
            if (row.source_participant < 0 ||
                row.destination_participant < 0 || row.layer < 0 ||
                row.source_participant == row.destination_participant ||
                static_cast<std::size_t>(row.source_participant) >=
                    participant_count ||
                static_cast<std::size_t>(row.destination_participant) >=
                    participant_count ||
                static_cast<std::size_t>(row.layer) >= layer_count ||
                row.transfer_and_repack_ns == 0u)
            {
                throw std::invalid_argument(
                    "certified device economy migration profile contains an invalid coordinate or zero transfer price");
            }
            const std::size_t offset =
                (static_cast<std::size_t>(row.source_participant) *
                     participant_count +
                 static_cast<std::size_t>(row.destination_participant)) *
                    layer_count +
                static_cast<std::size_t>(row.layer);
            if (migration_seen[offset])
            {
                throw std::invalid_argument(
                    "certified device economy migration profile repeats a directed coordinate");
            }
            migration[offset] = {
                .transfer_and_repack_ns = row.transfer_and_repack_ns,
                .inference_interference_ns =
                    row.inference_interference_ns,
            };
            migration_seen[offset] = true;
        }
        if (std::find(
                migration_seen.begin(), migration_seen.end(), false) !=
            migration_seen.end())
        {
            throw std::invalid_argument(
                "certified device economy migration profile omitted a directed coordinate");
        }

        economy->active_source_bits = active_source_bits;
        economy->service_identity_fingerprint =
            economyIdentityFingerprint(profiles.service->identity);
        economy->migration_identity_fingerprint =
            economyIdentityFingerprint(profiles.migration->identity);
        economy->publication_generation = 1u;
        economy->historical_window_weight =
            profiles.policy.historical_window_weight;
        economy->current_window_weight =
            profiles.policy.current_window_weight;
        economy->minimum_residency_generations =
            profiles.policy.minimum_residency_generations;
        economy->payoff_horizon_tokens =
            profiles.policy.payoff_horizon_tokens;
        economy->minimum_net_benefit_ns =
            profiles.policy.minimum_net_benefit_ns;

        /* This is the sole host-to-device evidence edge. All placement and
         * epoch state remains untouched; the leader kernel acquires `state`
         * before consuming any ordinary array store above. */
        storeRelease(
            economy->state,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerEconomyState::Ready));

        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "device_economy_profile_publications",
            1.0,
            "maintenance",
            {},
            {{"service_profile", profiles.service->identity},
             {"migration_profile", profiles.migration->identity},
             {"policy_owner", "device"},
             {"evidence_owner", "host"},
             {"blocking_inference", "false"}});
    }

    bool MoEOverlayNodeLocalDeviceControllerFabric::
        economyProfilesPublished() const noexcept
    {
        if (!mapping_lifetime_ || !mapping_lifetime_->base ||
            mapping_lifetime_->base == MAP_FAILED || !layout_.valid())
        {
            return false;
        }
        auto *const economy = at<
            MoEOverlayDeviceControllerEconomyHeader>(
            static_cast<std::byte *>(mapping_lifetime_->base),
            static_cast<std::size_t>(
                layout_.header.economy_header_offset));
        return loadAcquire(economy->state) ==
               static_cast<std::uint32_t>(
                   MoEOverlayDeviceControllerEconomyState::Ready);
    }

    void MoEOverlayNodeLocalDeviceControllerFabric::
        publishLocalDemandHistogramRebase()
    {
        if (!economyProfilesPublished() || !mapping_lifetime_ ||
            !mapping_lifetime_->base ||
            mapping_lifetime_->base == MAP_FAILED ||
            !config_.mpi_ctx)
        {
            throw std::logic_error(
                "device controller histogram rebase requires a published economy profile and live mapped fabric");
        }

        const int local_rank = config_.mpi_ctx->rank();
        const auto position = std::lower_bound(
            participating_world_ranks_.begin(),
            participating_world_ranks_.end(),
            local_rank);
        if (position == participating_world_ranks_.end() ||
            *position != local_rank)
        {
            throw std::logic_error(
                "device controller histogram rebase rank is absent from the frozen rendezvous");
        }

        auto *const setup = static_cast<
            MoEOverlayDeviceControllerFabricSetupHeader *>(
            mapping_lifetime_->base);
        const auto index = static_cast<std::size_t>(
            position - participating_world_ranks_.begin());
        if (loadAcquire(setup->demand_histogram_rebase_ready[index]) != 0u)
        {
            throw std::logic_error(
                "device controller histogram rebase may be published exactly once per rank");
        }

        // The device terminal events were acquired before this release store.
        // Peers therefore cannot observe the acknowledgement before both
        // process-local phase baselines are durable on every local GPU.
        storeRelease(
            setup->demand_histogram_rebase_ready[index],
            kSetupPublication);
    }

    bool MoEOverlayNodeLocalDeviceControllerFabric::
        demandHistogramsRebased() const noexcept
    {
        if (!mapping_lifetime_ || !mapping_lifetime_->base ||
            mapping_lifetime_->base == MAP_FAILED ||
            participating_world_ranks_.empty())
        {
            return false;
        }
        const auto *const setup = static_cast<const
            MoEOverlayDeviceControllerFabricSetupHeader *>(
            mapping_lifetime_->base);
        for (std::size_t index = 0u;
             index < participating_world_ranks_.size();
             ++index)
        {
            if (loadAcquire(
                    setup->demand_histogram_rebase_ready[index]) !=
                kSetupPublication)
            {
                return false;
            }
        }
        return true;
    }

    bool MoEOverlayNodeLocalDeviceControllerFabric::
        trySnapshotServiceTelemetry(
            int participant_id,
            std::vector<MoEOverlayParticipantLayerServiceTotals> *output,
            std::uint64_t *generation) const noexcept
    {
        if (generation)
            *generation = 0u;
        if (!output)
            return false;
        output->clear();

        try
        {
            if (!mapping_lifetime_ || !mapping_lifetime_->base ||
                mapping_lifetime_->base == MAP_FAILED || !layout_.valid() ||
                !config_.topology || !config_.mpi_ctx || participant_id < 0 ||
                static_cast<std::size_t>(participant_id) >=
                    config_.topology->participants.size())
            {
                return false;
            }

            const auto &participant = config_.topology->participants[
                static_cast<std::size_t>(participant_id)];
            if (participant.participant_id != participant_id ||
                participant.world_rank != config_.mpi_ctx->rank())
            {
                return false;
            }
            const auto *const topology_group =
                config_.topology->groupForParticipant(participant_id);
            if (!topology_group || topology_group->group_id < 0 ||
                static_cast<std::size_t>(topology_group->group_id) >=
                    layout_.groups.size())
            {
                return false;
            }
            const auto member = std::find(
                topology_group->participant_ids.begin(),
                topology_group->participant_ids.end(),
                participant_id);
            if (member == topology_group->participant_ids.end())
                return false;

            const auto &group = layout_.groups[
                static_cast<std::size_t>(topology_group->group_id)];
            const std::size_t member_index = static_cast<std::size_t>(
                member - topology_group->participant_ids.begin());
            auto *const base = static_cast<std::byte *>(
                mapping_lifetime_->base);
            auto *const publication = at<
                MoEOverlayDeviceServiceTelemetryPublicationHeader>(
                base,
                checkedAdd(
                    static_cast<std::size_t>(
                        group.service_telemetry_offset),
                    checkedMultiply(
                        member_index,
                        static_cast<std::size_t>(
                            group.service_telemetry_stride_bytes),
                        "service telemetry snapshot member offset"),
                    "service telemetry snapshot address"));
            return trySnapshotMoEOverlayServiceTelemetryPublication(
                publication,
                participant_id,
                layout_.header.num_layers,
                output,
                generation);
        }
        catch (...)
        {
            output->clear();
            if (generation)
                *generation = 0u;
            return false;
        }
    }

    bool MoEOverlayDeviceControllerTransportBinding::valid() const noexcept
    {
        if (!(group_id >= 0 && root_world_rank >= 0 &&
               topology_fingerprint != 0u && command_capacity != 0u &&
               layout && controller && command && command_entries &&
               group_record_count > 0u &&
               group_record_count <=
                   kMoEOverlayDeviceControllerFabricMaxParticipants &&
               participant_records && participant_record_count > 0u &&
               participant_record_count <=
                   kMoEOverlayDeviceControllerFabricMaxParticipants &&
               transport && lifetime &&
               layout->magic == kMoEOverlayDeviceControllerFabricMagic &&
               layout->version == kMoEOverlayDeviceControllerFabricVersion &&
               layout->topology_fingerprint == topology_fingerprint &&
               layout->command_capacity == command_capacity &&
               transport->magic == kMoEOverlayDeviceControllerFabricMagic &&
               transport->version ==
                   kMoEOverlayDeviceControllerFabricVersion &&
               transport->group_id == static_cast<std::uint32_t>(group_id) &&
               transport->topology_fingerprint == topology_fingerprint))
        {
            return false;
        }
        for (std::uint32_t index = 0u; index < group_record_count; ++index)
        {
            const auto *record = group_records[index];
            const auto *participants = topology_participant_records[index];
            const auto participant_count =
                topology_participant_record_counts[index];
            if (!record || record->magic != kMoEOverlayDeviceControllerMagic ||
                record->version != kMoEOverlayDeviceControllerVersion ||
                record->group_id != index ||
                record->topology_fingerprint != topology_fingerprint ||
                !participants || participant_count == 0u ||
                participant_count >
                    kMoEOverlayDeviceControllerFabricMaxParticipants)
            {
                return false;
            }
        }
        return true;
    }

    MoEOverlayDeviceControllerTransportBinding
    MoEOverlayNodeLocalDeviceControllerFabric::transportBinding(
        int group_id) const
    {
        if (!mapping_lifetime_ || !mapping_lifetime_->base || group_id < 0 ||
            static_cast<std::size_t>(group_id) >= layout_.groups.size())
        {
            throw std::out_of_range(
                "device controller transport binding is unavailable");
        }
        const auto &group = layout_.groups[static_cast<std::size_t>(group_id)];
        if (group.group_id != static_cast<std::uint32_t>(group_id) ||
            group.root_world_rank != config_.mpi_ctx->rank())
        {
            throw std::out_of_range(
                "device controller transport binding belongs to another rank");
        }

        auto *const base = static_cast<std::byte *>(mapping_lifetime_->base);
        MoEOverlayDeviceControllerTransportBinding binding{
            .group_id = group_id,
            .root_world_rank = group.root_world_rank,
            .topology_fingerprint =
                config_.topology->topology_fingerprint,
            .command_capacity = config_.command_capacity,
            .layout = at<MoEOverlayDeviceControllerFabricLayoutHeader>(
                base, layout_.leader_owned_begin),
            .controller = at<MoEOverlayDeviceControllerSharedHeader>(
                base,
                static_cast<std::size_t>(
                    layout_.header.controller_header_offset)),
            .command = at<MoEOverlayDeviceControllerCommandHeader>(
                base,
                static_cast<std::size_t>(
                    layout_.header.command_header_offset)),
            .command_entries = at<MoEOverlayDeviceMovementCommand>(
                base,
                static_cast<std::size_t>(
                    layout_.header.command_entries_offset)),
            .group_record_count = layout_.header.group_count,
            .participant_records = at<
                MoEOverlayDeviceControllerParticipantRecord>(
                base,
                static_cast<std::size_t>(
                    group.participant_records_offset)),
            .participant_record_count = group.participant_count,
            .transport = at<MoEOverlayDeviceControllerTransportRecord>(
                base,
                static_cast<std::size_t>(group.transport_record_offset)),
            .lifetime = std::static_pointer_cast<const void>(
                mapping_lifetime_),
        };
        for (std::uint32_t index = 0u;
             index < binding.group_record_count;
             ++index)
        {
            binding.group_records[index] = at<
                MoEOverlayDeviceControllerGroupRecord>(
                base,
                static_cast<std::size_t>(
                    layout_.groups[index].group_record_offset));
            binding.topology_participant_records[index] = at<
                MoEOverlayDeviceControllerParticipantRecord>(
                base,
                static_cast<std::size_t>(
                    layout_.groups[index].participant_records_offset));
            binding.topology_participant_record_counts[index] =
                layout_.groups[index].participant_record_count;
        }
        if (!binding.valid())
        {
            throw std::logic_error(
                "device controller transport binding failed immutable identity validation");
        }
        return binding;
    }
} // namespace llaminar2
