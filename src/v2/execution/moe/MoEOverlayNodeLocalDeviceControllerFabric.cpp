/**
 * @file MoEOverlayNodeLocalDeviceControllerFabric.cpp
 * @brief POSIX mapping, NUMA placement, and GPU alias setup for controller RCU.
 *
 * This file executes before graph construction. It may wait for local MPI
 * peers, but it never observes live histogram data or participates in an
 * inference transaction. Once construction returns, CUDA/HIP kernels are the
 * only writers of controller lifecycle and policy records.
 */

#include "utils/HostPageSize.h"
#include "MoEOverlayNodeLocalDeviceControllerFabric.h"
#include "MoEOverlayPreparedTargetService.h"
#include "MoEOverlaySharedBackingStartup.h"

#include "MoEOverlayEconomyProfileComposer.h"
#include "MoEOverlayServiceTelemetryPublication.h"

#include "../../utils/FNV1a.h"
#include "interfaces/IMPIContext.h"
#include "interfaces/IMPITopology.h"
#include "transfer/TransferEngine.h"
#include "utils/PerfStatsCollector.h"
#include "utils/Logger.h"

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
#include <set>
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
                // Every populated tier owns at least one original member.
                // This is a structural extent check, not a hardware ceiling.
                if (tier >= topology.participants.size())
                    throw std::invalid_argument("device controller tier index exceeds its participant extent");
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
            std::uint64_t topology_fingerprint,
            std::int32_t first_model_layer,
            std::uint32_t num_layers)
        {
            std::ostringstream name;
            name << "llaminar_moectrl_" << std::hex << std::setw(16)
                 << std::setfill('0') << node_namespace << '_'
                 << std::setw(16) << topology_fingerprint
                 << "_layers_" << first_model_layer << '_' << num_layers;
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
            std::uint32_t num_layers,
            std::int32_t first_model_layer) noexcept
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
            mix(static_cast<std::uint32_t>(first_model_layer));
            for (std::uint32_t layer = 0u; layer < num_layers; ++layer)
                mix(bytes.empty() ? 0u : bytes[layer]);
            return hash == 0u ? 1u : hash;
        }
    } // namespace

    bool MoEOverlayDeviceControllerFabricLayout::containsModelLayer(int layer) const noexcept
    {
        return header.first_model_layer >= 0 && header.num_layers > 0u &&
            header.num_layers <= static_cast<std::uint32_t>(
                std::numeric_limits<int>::max() - header.first_model_layer) &&
            layer >= header.first_model_layer &&
            static_cast<std::uint32_t>(layer - header.first_model_layer) < header.num_layers;
    }

    std::uint32_t MoEOverlayDeviceControllerFabricLayout::storageIndexForModelLayer(int layer) const
    {
        if (!containsModelLayer(layer))
            throw std::out_of_range("MoE controller evidence layer is outside the mapped stage");
        return static_cast<std::uint32_t>(layer - header.first_model_layer);
    }

    struct MoEOverlayNodeLocalDeviceControllerFabric::MappingLifetime
    {
        /** Unmap and close only after every GPU registration releases us. */
        ~MappingLifetime()
        {
            if (base && base != MAP_FAILED && bytes != 0u)
                (void)::munmap(base, bytes);
        }

        MPISessionOwner::SharedFile file;
        void *base = nullptr;
        std::size_t bytes = 0u;
    };

    bool MoEOverlayDeviceControllerFabricLayout::valid() const noexcept
    {
        // Prove complete bounds before evaluating any offset+size ordering
        // expression below. A corrupt count cannot wrap into a valid page.
        if (!header.participant_count || !header.logical_target_count ||
            header.logical_target_count > header.participant_count ||
            header.logical_target_count != logical_targets.size() ||
            header.demand_publisher_participant_id >= header.participant_count ||
            header.demand_publisher_group_id >= header.group_count ||
            !header.num_layers || !header.num_experts ||
            header.num_layers > kMoEOverlayDeviceControllerFabricMaxLayers ||
            header.num_experts > kMoEOverlayDeviceControllerFabricMaxExperts ||
            std::uint64_t(header.participant_count) * header.participant_count >
                UINT64_MAX / header.num_layers ||
            !moeOverlayExtentContains<MoEOverlayDeviceControllerFabricLayoutHeader>(leader_owned_end, leader_owned_begin, 1u) ||
            !moeOverlayExtentContains<MoEOverlayDeviceControllerParticipantMetadata>(leader_owned_end, header.participant_metadata_offset, header.participant_count) ||
            !moeOverlayExtentContains<MoEOverlayDeviceControllerFabricGroupLayout>(leader_owned_end, header.group_layout_offset, groups.size()) ||
            !moeOverlayExtentContains<MoEOverlayDeviceControllerLogicalTargetMetadata>(leader_owned_end,
                header.logical_target_metadata_offset, header.logical_target_count) ||
            !moeOverlayExtentContains<MoEOverlayDeviceControllerSharedHeader>(leader_owned_end, header.controller_header_offset, 1u) ||
            !moeOverlayExtentContains<MoEOverlayDeviceControllerCommandHeader>(leader_owned_end, header.command_header_offset, 1u) ||
            !moeOverlayExtentContains<MoEOverlayDeviceMovementCommand>(leader_owned_end, header.command_entries_offset, header.command_capacity) ||
            !moeOverlayExtentContains<std::uint64_t>(leader_owned_end, header.payload_bytes_per_layer_offset, header.num_layers) ||
            !moeOverlayExtentContains<MoEOverlayRouteTarget>(leader_owned_end, header.initial_route_targets_offset, header.initial_route_targets_words) ||
            !moeOverlayExtentContains<std::uint64_t>(leader_owned_end, header.demand_history_offset, header.demand_history_words) ||
            !moeOverlayExtentContains<MoEOverlayDeviceControllerEconomyHeader>(leader_owned_end, header.economy_header_offset, 1u) ||
            !moeOverlayExtentContains<std::uint64_t>(leader_owned_end, header.economy_service_cost_offset, header.economy_service_cost_words) ||
            !moeOverlayExtentContains<MoEOverlayDeviceControllerMigrationCost>(leader_owned_end, header.economy_migration_cost_offset, header.economy_migration_cost_entries) ||
            !moeOverlayExtentContains<std::uint64_t>(leader_owned_end, header.economy_last_moved_offset, header.economy_last_moved_words) ||
            header.participant_metadata_offset < leader_owned_begin + sizeof(header) ||
            header.logical_target_metadata_offset < header.participant_metadata_offset + std::uint64_t(header.participant_count) * sizeof(MoEOverlayDeviceControllerParticipantMetadata) ||
            header.group_layout_offset < header.logical_target_metadata_offset + std::uint64_t(header.logical_target_count) * sizeof(MoEOverlayDeviceControllerLogicalTargetMetadata) ||
            header.command_entries_offset < header.command_header_offset + sizeof(MoEOverlayDeviceControllerCommandHeader) ||
            header.payload_bytes_per_layer_offset < header.command_entries_offset + std::uint64_t(header.command_capacity) * sizeof(MoEOverlayDeviceMovementCommand))
            return false;
        if (page_bytes == 0u ||
            (page_bytes & (page_bytes - 1u)) != 0u ||
            mapping_bytes == 0u || mapping_bytes % page_bytes != 0u ||
            setup_header_offset != 0u ||
            setup.storage_bytes != leader_owned_begin || setup.participating_rank_count == 0u ||
            setup.participating_rank_count > header.participant_count ||
            setup.participating_world_ranks_offset < sizeof(setup) ||
            !moeOverlayExtentContains<std::int32_t>(leader_owned_begin, setup.participating_world_ranks_offset, setup.participating_rank_count) ||
            !moeOverlayExtentContains<std::uint64_t>(leader_owned_begin, setup.first_touch_ready_offset, setup.participating_rank_count) ||
            !moeOverlayExtentContains<std::uint64_t>(leader_owned_begin, setup.registration_ready_offset, setup.participating_rank_count) ||
            !moeOverlayExtentContains<std::uint64_t>(leader_owned_begin, setup.demand_histogram_rebase_ready_offset, setup.participating_rank_count) ||
            setup.first_touch_ready_offset < setup.participating_world_ranks_offset + std::uint64_t(setup.participating_rank_count) * sizeof(std::int32_t) ||
            setup.registration_ready_offset < setup.first_touch_ready_offset + std::uint64_t(setup.participating_rank_count) * sizeof(std::uint64_t) ||
            setup.demand_histogram_rebase_ready_offset < setup.registration_ready_offset + std::uint64_t(setup.participating_rank_count) * sizeof(std::uint64_t) ||
            leader_owned_begin < page_bytes ||
            leader_owned_begin >= leader_owned_end ||
            leader_owned_end > mapping_bytes ||
            leader_owned_begin % page_bytes != 0u ||
            leader_owned_end % page_bytes != 0u || groups.empty() ||
            header.magic != kMoEOverlayDeviceControllerFabricMagic ||
            header.version != kMoEOverlayDeviceControllerFabricVersion ||
            header.mapping_bytes != mapping_bytes ||
            header.group_count != groups.size() ||
            header.participant_count == 0u || header.num_layers == 0u ||
            header.first_model_layer < 0 ||
            header.num_layers > static_cast<std::uint32_t>(
                std::numeric_limits<int>::max() - header.first_model_layer) ||
            header.num_experts == 0u || header.command_capacity == 0u ||
            header.routed_experts_per_token == 0u ||
            header.routed_experts_per_token > header.num_experts ||
            header.tier_count == 0u ||
            header.tier_count > header.logical_target_count ||
            header.payload_bytes_per_layer_offset < leader_owned_begin ||
            header.payload_bytes_per_layer_offset +
                    header.num_layers * sizeof(std::uint64_t) >
                leader_owned_end ||
            header.initial_route_targets_offset <
                header.payload_bytes_per_layer_offset +
                    header.num_layers * sizeof(std::uint64_t) ||
            header.initial_route_targets_offset %
                    alignof(std::uint32_t) !=
                0u ||
            header.initial_route_targets_words !=
                static_cast<std::uint64_t>(header.num_layers) *
                    header.num_experts ||
            header.initial_route_targets_offset +
                    header.initial_route_targets_words *
                        sizeof(std::uint32_t) >
                leader_owned_end ||
            header.demand_history_offset < leader_owned_begin ||
            header.demand_history_offset <
                header.initial_route_targets_offset +
                    header.initial_route_targets_words *
                        sizeof(std::uint32_t) ||
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
                static_cast<std::uint64_t>(header.logical_target_count) *
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
                static_cast<std::uint64_t>(header.logical_target_count) *
                    header.logical_target_count * header.num_layers ||
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
            header.inference_epoch_storage_bytes < sizeof(MoEOverlayDeviceControllerInferenceEpochRecord) ||
            !moeOverlayExtentContains<std::byte>(leader_owned_end, header.inference_epoch_record_offset,
                header.inference_epoch_storage_bytes) ||
            header.command_header_offset < header.inference_epoch_record_offset + header.inference_epoch_storage_bytes ||
            header.minimum_window_activations == 0u ||
            header.maximum_cycles_per_wave == 0u ||
            header.payload_geometry_fingerprint == 0u)
        {
            return false;
        }

        std::uint64_t logical_members = 0u;
        for (std::size_t vertex = 0; vertex < logical_targets.size(); ++vertex)
        {
            const auto &target = logical_targets[vertex];
            if (target.group_id >= groups.size() || target.target.isAbsent() || target.reserved) return false;
            const auto &group = groups[target.group_id];
            if (target.tier_index != group.tier_index || target.tier_priority != group.tier_priority ||
                !target.member_count || target.first_group_member >= group.participant_count ||
                target.member_count > group.participant_count - target.first_group_member ||
                (target.target.isCohort()
                    ? (target.target.word() != group.cohort_target.word() || target.first_group_member || target.member_count != group.participant_count)
                    : (!group.cohort_target.isAbsent() || target.member_count != 1u ||
                       static_cast<std::uint32_t>(target.target.word()) >= header.participant_count))) return false;
            for (std::size_t previous = 0; previous < vertex; ++previous)
            {
                const auto &other = logical_targets[previous];
                if (other.target.word() == target.target.word() || (other.group_id == target.group_id &&
                    other.first_group_member < target.first_group_member + target.member_count &&
                    target.first_group_member < other.first_group_member + other.member_count)) return false;
            }
            logical_members += target.member_count;
        }
        if (logical_members != header.participant_count) return false;
        std::size_t previous_end = leader_owned_end;
        std::uint64_t roster_frontier = header.group_layout_offset + groups.size() * sizeof(MoEOverlayDeviceControllerFabricGroupLayout);
        std::uint64_t members = 0u;
        std::uint32_t ranks = 0u;
        for (std::size_t index = 0; index < groups.size(); ++index)
        {
            const auto &group = groups[index];
            if (!group.cohort_target.isAbsent() &&
                (!group.cohort_target.isCohort() || !group.participant_count || group.tier_index < 0 ||
                 std::int64_t(group.cohort_target.word()) != -std::int64_t(group.tier_index) - 2 ||
                 group.intra_group_transport != static_cast<std::uint32_t>(group.participant_count == 1u
                    ? MoEOverlayDeviceControllerIntraGroupTransport::SingleParticipant
                    : MoEOverlayDeviceControllerIntraGroupTransport::NativeCollective)))
                return false;
            bool first_rank = true;
            for (std::size_t previous = 0u; previous < index; ++previous)
                first_rank = first_rank && groups[previous].root_world_rank != group.root_world_rank;
            if (first_rank) ++ranks;
            if (group.participant_ids_offset < roster_frontier ||
                !moeOverlayExtentContains<std::uint32_t>(header.controller_header_offset, group.participant_ids_offset, group.participant_count) ||
                !moeOverlayExtentContains<MoEOverlayDeviceControllerParticipantRecord>(group.owned_page_end, group.participant_records_offset, group.participant_record_count) ||
                !moeOverlayExtentContains<MoEOverlayDeviceControllerGroupRecord>(group.owned_page_end, group.group_record_offset, 1u) ||
                !moeOverlayExtentContains<MoEOverlayDeviceControllerTransportRecord>(group.owned_page_end, group.transport_record_offset, 1u) ||
                !moeOverlayExtentContains<std::uint64_t>(group.owned_page_end, group.collected_state_offset, group.collected_state_words) ||
                !moeOverlayExtentContains<std::byte>(group.owned_page_end, group.service_telemetry_offset, group.service_telemetry_bytes) ||
                (group.participant_count && group.service_telemetry_stride_bytes > UINT64_MAX / group.participant_count) ||
                group.group_record_offset < group.participant_records_offset + group.participant_record_count * sizeof(MoEOverlayDeviceControllerParticipantRecord) ||
                group.transport_record_offset < group.group_record_offset + sizeof(MoEOverlayDeviceControllerGroupRecord) ||
                group.collected_state_offset < group.transport_record_offset + sizeof(MoEOverlayDeviceControllerTransportRecord) ||
                group.service_telemetry_offset < group.collected_state_offset + group.collected_state_words * sizeof(std::uint64_t))
                return false;
            roster_frontier = group.participant_ids_offset + std::uint64_t(group.participant_count) * sizeof(std::uint32_t);
            members += group.participant_count;
            if (group.group_id != index || group.participant_count == 0u ||
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
                group.collected_state_words !=
                    (static_cast<std::uint64_t>(group.participant_count) +
                        (group.group_id == header.demand_publisher_group_id ? 2u : 0u)) *
                        header.num_layers * header.num_experts ||
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
        return previous_end == mapping_bytes && members == header.participant_count &&
               ranks == setup.participating_rank_count;
    }

    bool MoEOverlayDeviceControllerParticipantBinding::valid() const noexcept
    {
        return device.is_gpu() && participant_id >= 0 && group_id >= 0 &&
               capture_metadata.topology_fingerprint != 0u &&
               capture_metadata.group_collected_state_words != 0u &&
               mapped_base_device && mapped_bytes != 0u && layout &&
               participants && groups && logical_targets && controller &&
               inference_epoch_record && inference_epoch_storage_bytes != 0u &&
               (!inference_epoch_member || inference_epoch_arrival) && command &&
               command_entries && payload_bytes_per_layer &&
               initial_route_targets && demand_history &&
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
        // A registered GPU alias need not be CPU-addressable. All scalar
        // arguments come from the frozen host layout, never from an alias
        // dereference or a copy of mutable device controller state.
        return {
            .mapped_base = mapped_base_device,
            .mapped_bytes = mapped_bytes,
            .layout = layout,
            .participants = participants,
            .groups = groups,
            .logical_targets = logical_targets,
            .controller = controller,
            .inference_epoch_record = inference_epoch_record,
            .command = command,
            .command_entries = command_entries,
            .payload_bytes_per_layer = payload_bytes_per_layer,
            .initial_route_targets = initial_route_targets,
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
            .topology_fingerprint = capture_metadata.topology_fingerprint,
            .participant_id = static_cast<std::uint32_t>(participant_id),
            .group_id = static_cast<std::uint32_t>(group_id),
            .role_flags = capture_metadata.role_flags,
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
        std::uint32_t routed_experts_per_token,
        std::int32_t first_model_layer)
    {
        if (!topology.valid() || num_layers == 0u || num_experts == 0u ||
            first_model_layer < 0 ||
            num_layers > static_cast<std::uint32_t>(std::numeric_limits<int>::max() - first_model_layer) ||
            num_layers > kMoEOverlayDeviceControllerFabricMaxLayers ||
            num_experts > kMoEOverlayDeviceControllerFabricMaxExperts ||
            command_capacity == 0u || routed_experts_per_token == 0u ||
            routed_experts_per_token > num_experts ||
            topology.participants.size() > std::numeric_limits<std::uint32_t>::max() ||
            topology.groups.size() > std::numeric_limits<std::uint32_t>::max() ||
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
        result.logical_targets = topology.logicalTargets();
        const auto target_count = result.logical_targets.size();
        result.page_bytes = page_bytes;
        result.setup_header_offset = 0u;
        std::set<int> rank_roster;
        for (const auto &group : topology.groups) rank_roster.insert(group.root_world_rank);
        const auto ranks = rank_roster.size();
        result.setup.participating_rank_count = static_cast<std::uint32_t>(ranks);
        std::size_t cursor = sizeof(MoEOverlayDeviceControllerFabricSetupHeader);
        result.setup.participating_world_ranks_offset = cursor;
        cursor = checkedAdd(cursor, checkedMultiply(ranks, sizeof(std::int32_t), "setup rank roster"), "setup rank roster");
        const auto add_ready_lanes = [&](std::uint64_t &offset) {
            cursor = checkedAlignUp(cursor, alignof(std::uint64_t), "setup publication lanes");
            offset = cursor;
            cursor = checkedAdd(cursor, checkedMultiply(ranks, sizeof(std::uint64_t), "setup publication lanes"), "setup publication lanes");
        };
        add_ready_lanes(result.setup.first_touch_ready_offset);
        add_ready_lanes(result.setup.registration_ready_offset);
        add_ready_lanes(result.setup.demand_histogram_rebase_ready_offset);
        result.leader_owned_begin = checkedAlignUp(cursor, page_bytes, "complete setup region");
        result.setup.storage_bytes = result.leader_owned_begin;
        cursor = result.leader_owned_begin;

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
        cursor = checkedAlignUp(cursor, alignof(MoEOverlayDeviceControllerLogicalTargetMetadata), "logical target metadata");
        const std::size_t logical_target_metadata_offset = cursor;
        cursor = checkedAdd(cursor, checkedMultiply(target_count,
            sizeof(MoEOverlayDeviceControllerLogicalTargetMetadata), "logical target metadata bytes"), "logical target metadata");
        cursor = checkedAlignUp(cursor, 64u, "group layout");
        const std::size_t group_layout_offset = cursor;
        cursor = checkedAdd(
            cursor,
            checkedMultiply(
                topology.groups.size(),
                sizeof(MoEOverlayDeviceControllerFabricGroupLayout),
                "group layout bytes"),
            "group layout");
        cursor = checkedAlignUp(cursor, alignof(std::uint32_t), "group member rosters");
        const std::size_t group_member_rosters_offset = cursor;
        cursor = checkedAdd(cursor, checkedMultiply(topology.participants.size(), sizeof(std::uint32_t),
            "group member roster bytes"), "group member rosters");
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
            MoEOverlayDeviceControllerInferenceEpochRecord::bytesForMembers(
                static_cast<std::uint32_t>(topology.groups.at(topology.leader_group_id).participant_ids.size())),
            "inference epoch record and exact member tail");
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
        cursor = checkedAlignUp(cursor, 64u, "initial prepared owner table");
        const std::size_t initial_route_targets_offset = cursor;
        const std::size_t initial_route_targets_words = checkedMultiply(
            num_layers,
            num_experts,
            "initial prepared owner words");
        cursor = checkedAdd(
            cursor,
            checkedMultiply(
                initial_route_targets_words,
                sizeof(std::uint32_t),
                "initial prepared owner bytes"),
            "initial prepared owner table");
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
            target_count,
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
            target_count,
            checkedMultiply(
                target_count,
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
        std::size_t roster_cursor = group_member_rosters_offset;
        for (const auto &topology_group : topology.groups)
        {
            if (!topology_group.valid())
            {
                throw std::invalid_argument(
                    "device controller fabric group has incomplete immutable membership");
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
            group.cohort_target = topology_group.cohort_target;
            group.intra_group_transport = static_cast<std::uint32_t>(
                topology_group.intra_group_transport);
            group.inter_group_transport = static_cast<std::uint32_t>(
                topology_group.inter_group_transport);
            group.participant_ids_offset = roster_cursor;
            roster_cursor = checkedAdd(roster_cursor, checkedMultiply(group.participant_count,
                sizeof(std::uint32_t), "group roster bytes"), "group roster frontier");

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
            std::size_t state_words = 0u;
            for (const auto participant : topology_group.participant_ids)
                state_words = checkedAdd(state_words, static_cast<std::size_t>(moeOverlaySnapshotMemberWords(
                    num_layers, num_experts, static_cast<std::uint32_t>(participant),
                    static_cast<std::uint32_t>(topology.leader_participant_id))), "live group member state words");
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
            .inference_epoch_storage_bytes = MoEOverlayDeviceControllerInferenceEpochRecord::bytesForMembers(
                static_cast<std::uint32_t>(topology.groups.at(topology.leader_group_id).participant_ids.size())),
            .command_header_offset = command_header_offset,
            .command_entries_offset = command_entries_offset,
            .payload_bytes_per_layer_offset =
                payload_bytes_per_layer_offset,
            .initial_route_targets_offset =
                initial_route_targets_offset,
            .initial_route_targets_words =
                initial_route_targets_words,
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
                payload_bytes_per_layer, num_layers, first_model_layer),
            .routed_experts_per_token = routed_experts_per_token,
            .first_model_layer = first_model_layer,
            .logical_target_count = static_cast<std::uint32_t>(target_count),
            .demand_publisher_participant_id = static_cast<std::uint32_t>(topology.leader_participant_id),
            .demand_publisher_group_id = static_cast<std::uint32_t>(topology.leader_group_id),
            .logical_target_metadata_offset = logical_target_metadata_offset,
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
        const std::size_t initial_owner_words =
            static_cast<std::size_t>(config_.num_layers) *
            static_cast<std::size_t>(config_.num_experts);
        const bool dynamic_geometry =
            !config_.payload_bytes_per_layer.empty();
        if (!config_.mpi_ctx || !config_.topology ||
            !config_.topology->valid() || config_.num_layers == 0u ||
            config_.num_experts == 0u || config_.command_capacity == 0u ||
            config_.routed_experts_per_token == 0u ||
            config_.routed_experts_per_token > config_.num_experts ||
            config_.initial_durable_epoch == 0u ||
            (!config_.payload_bytes_per_layer.empty() &&
             config_.payload_bytes_per_layer.size() != config_.num_layers) ||
            (dynamic_geometry &&
             config_.initial_route_targets.size() !=
                 initial_owner_words) ||
            (!config_.initial_route_targets.empty() &&
             config_.initial_route_targets.size() !=
                 initial_owner_words) ||
            config_.minimum_window_activations == 0u ||
            config_.maximum_cycles_per_wave == 0u)
        {
            throw std::invalid_argument(
                "node-local device controller fabric requires complete topology and geometry");
        }
        if (std::any_of(
                config_.initial_route_targets.begin(),
                config_.initial_route_targets.end(),
                [this](MoEOverlayRouteTarget target)
                { return !config_.topology->groupForTarget(target); }))
        {
            throw std::invalid_argument(
                "node-local device controller fabric initial target table names a foreign original target");
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
            hostPageBytes(),
            config_.payload_bytes_per_layer,
            config_.minimum_window_activations,
            config_.maximum_cycles_per_wave,
            config_.dynamic_imbalance_threshold_per_mille,
            config_.dynamic_minimum_improvement_per_mille,
            config_.dynamic_maximum_cycles_per_layer,
            config_.dynamic_maximum_commands_per_wave,
            config_.routed_experts_per_token,
            config_.first_model_layer);

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
            config_.topology->topology_fingerprint,
            config_.first_model_layer,
            config_.num_layers);
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
            throw;
        }
    }

    MoEOverlayNodeLocalDeviceControllerFabric::
        ~MoEOverlayNodeLocalDeviceControllerFabric()
    {
        // POSIX unlink only removes the name; every already-open mapping stays
        // valid until its process-local driver registrations are released.
        if(creator_ && mapping_lifetime_)try {mapping_lifetime_->file.unlink();}
        catch(const std::exception &error) {LOG_WARN("MoE owned device fabric name retained for ancestor cleanup: "<<error.what());}
        mapped_region_.reset();
        mapping_lifetime_.reset();
    }

    void MoEOverlayNodeLocalDeviceControllerFabric::mapOrAttach()
    {
        const int local_rank = config_.mpi_ctx->rank();
        creator_ = local_rank == config_.topology->leader_world_rank;
        auto lifetime=std::make_shared<MappingLifetime>();
        lifetime->file=establishMoEOverlaySharedBacking(*config_.mpi_ctx,{
            .kind=MoEOverlaySharedBackingKind::DeviceControllerFabric,
            .node_namespace=config_.mpi_ctx->topology()->node_shared_memory_namespace(),
            .generation=config_.initial_durable_epoch,
            .fingerprint_low=config_.topology->topology_fingerprint,
            .fingerprint_high=static_cast<std::uint64_t>(static_cast<std::uint32_t>(config_.first_model_layer))<<32u | config_.num_layers},
            config_.topology->leader_world_rank,participating_world_ranks_,channel_name_,layout_.mapping_bytes);
        lifetime->file.requireExtent(layout_.mapping_bytes);
        void *const base=::mmap(nullptr,layout_.mapping_bytes,PROT_READ|PROT_WRITE,MAP_SHARED,lifetime->file.descriptor(),0);
        if(base==MAP_FAILED)throw std::runtime_error("device controller fabric mmap failed: "+std::string(std::strerror(errno)));
        lifetime->base=base;lifetime->bytes=layout_.mapping_bytes;
        mapping_lifetime_=std::move(lifetime);

#if defined(MADV_HUGEPAGE)
        // Group regions remain page isolated even when the kernel chooses a
        // huge-page backing; first-touch still determines their NUMA home.
        (void)::madvise(base, layout_.mapping_bytes, MADV_HUGEPAGE);
#endif

        auto *const setup = static_cast<
            MoEOverlayDeviceControllerFabricSetupHeader *>(base);
        if (creator_)
        {
            std::memset(base, 0, layout_.leader_owned_begin);
            *setup = layout_.setup;
            setup->topology_fingerprint =
                config_.topology->topology_fingerprint;
            setup->node_namespace =
                config_.mpi_ctx->topology()->node_shared_memory_namespace();
            setup->mapping_bytes = layout_.mapping_bytes;
            setup->num_layers = config_.num_layers;
            setup->first_model_layer = config_.first_model_layer;
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
                moeOverlayExtentArray<std::int32_t>(setup, layout_.leader_owned_begin, setup->participating_world_ranks_offset, setup->participating_rank_count)[index] =
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
            setup->first_model_layer != config_.first_model_layer ||
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
            setup->storage_bytes != layout_.setup.storage_bytes ||
            setup->participating_world_ranks_offset != layout_.setup.participating_world_ranks_offset ||
            setup->first_touch_ready_offset != layout_.setup.first_touch_ready_offset ||
            setup->registration_ready_offset != layout_.setup.registration_ready_offset ||
            setup->demand_histogram_rebase_ready_offset != layout_.setup.demand_histogram_rebase_ready_offset ||
            !std::equal(
                participating_world_ranks_.begin(),
                participating_world_ranks_.end(),
                moeOverlayExtentArray<std::int32_t>(setup, layout_.leader_owned_begin, setup->participating_world_ranks_offset, setup->participating_rank_count)))
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
                    config_.topology->groups.at(group.group_id).participant_ids[member];
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
                    config_.topology->groups.at(group.group_id).participant_ids[member]);
                service->layer_count = config_.num_layers;
                service->first_model_layer = config_.first_model_layer;
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
        storeRelease(moeOverlayExtentArray<std::uint64_t>(setup, layout_.leader_owned_begin, setup->first_touch_ready_offset, setup->participating_rank_count)[index], kSetupPublication);
        waitForRankPublications(
            moeOverlayExtentArray<std::uint64_t>(setup, layout_.leader_owned_begin, setup->first_touch_ready_offset, setup->participating_rank_count),
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
            std::copy(layout_.logical_targets.begin(), layout_.logical_targets.end(),
                at<MoEOverlayDeviceControllerLogicalTargetMetadata>(base,
                    static_cast<std::size_t>(layout_.header.logical_target_metadata_offset)));

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
                static_cast<std::size_t>(config_.topology->leader_group_id));
            auto *const inference_epoch = at<MoEOverlayDeviceControllerInferenceEpochRecord>(base,
                static_cast<std::size_t>(layout_.header.inference_epoch_record_offset));
            *inference_epoch = MoEOverlayDeviceControllerInferenceEpochRecord{};
            inference_epoch->bindMemberExtent(static_cast<std::uint32_t>(inference_group.participant_ids.size()));
            inference_epoch->publisher_participant_id = static_cast<std::uint32_t>(config_.topology->leader_participant_id);
            inference_epoch->topology_fingerprint = config_.topology->topology_fingerprint;
            inference_epoch->epoch = config_.initial_durable_epoch;
            auto *epoch_ids = moeOverlayExtentArray<std::uint32_t>(inference_epoch,
                layout_.header.inference_epoch_storage_bytes, inference_epoch->member_ids_offset,
                inference_epoch->member_count);
            std::copy(inference_group.participant_ids.begin(), inference_group.participant_ids.end(), epoch_ids);
            std::sort(epoch_ids, epoch_ids + inference_epoch->member_count);
            for (const auto &group : layout_.groups)
            {
                auto *ids = moeOverlayExtentArray<std::uint32_t>(base, layout_.mapping_bytes,
                    group.participant_ids_offset, group.participant_count);
                const auto &original = config_.topology->groups.at(group.group_id).participant_ids;
                std::copy(original.begin(), original.end(), ids);
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
            auto *const initial_owners = at<MoEOverlayRouteTarget>(
                base,
                static_cast<std::size_t>(
                    layout_.header.initial_route_targets_offset));
            if (config_.initial_route_targets.empty())
            {
                std::fill_n(
                    initial_owners,
                    static_cast<std::size_t>(
                        layout_.header.initial_route_targets_words),
                    MoEOverlayRouteTarget{});
            }
            else
            {
                std::copy(
                    config_.initial_route_targets.begin(),
                    config_.initial_route_targets.end(),
                    initial_owners);
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
            economy->logical_target_count =
                layout_.header.logical_target_count;
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
        const auto *const published_initial_owners = at<MoEOverlayRouteTarget>(
            base,
            static_cast<std::size_t>(
                layout_.header.initial_route_targets_offset));
        const bool initial_owners_match =
            config_.initial_route_targets.empty() ||
            std::equal(
                config_.initial_route_targets.begin(),
                config_.initial_route_targets.end(),
                published_initial_owners, [](MoEOverlayRouteTarget a, MoEOverlayRouteTarget b) { return a.word() == b.word(); });
        auto expected_inference_ids = config_.topology->groups.at(
            static_cast<std::size_t>(config_.topology->leader_group_id)).participant_ids;
        std::sort(expected_inference_ids.begin(), expected_inference_ids.end());
        const auto *published_inference_ids = published_inference_epoch->memberIds(
            layout_.header.inference_epoch_storage_bytes);
        const bool inference_roster_matches = published_inference_ids &&
            published_inference_epoch->member_count == expected_inference_ids.size() &&
            std::equal(expected_inference_ids.begin(), expected_inference_ids.end(), published_inference_ids);
        const bool logical_roster_matches = std::equal(layout_.logical_targets.begin(), layout_.logical_targets.end(),
            at<MoEOverlayDeviceControllerLogicalTargetMetadata>(base,
                static_cast<std::size_t>(layout_.header.logical_target_metadata_offset)));
        if (published_layout->magic !=
                kMoEOverlayDeviceControllerFabricMagic ||
            published_layout->version !=
                kMoEOverlayDeviceControllerFabricVersion ||
            published_layout->first_model_layer != config_.first_model_layer ||
            published_layout->num_layers != config_.num_layers ||
            published_layout->inference_epoch_record_offset !=
                layout_.header.inference_epoch_record_offset ||
            published_layout->inference_epoch_storage_bytes != layout_.header.inference_epoch_storage_bytes ||
            published_layout->payload_bytes_per_layer_offset !=
                layout_.header.payload_bytes_per_layer_offset ||
            published_layout->initial_route_targets_offset !=
                layout_.header.initial_route_targets_offset ||
            published_layout->initial_route_targets_words !=
                layout_.header.initial_route_targets_words ||
            !initial_owners_match || !logical_roster_matches ||
            published_layout->logical_target_count != layout_.header.logical_target_count ||
            published_layout->logical_target_metadata_offset != layout_.header.logical_target_metadata_offset ||
            published_layout->demand_publisher_participant_id != layout_.header.demand_publisher_participant_id ||
            published_layout->demand_publisher_group_id != layout_.header.demand_publisher_group_id ||
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
            !inference_roster_matches ||
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
        storeRelease(moeOverlayExtentArray<std::uint64_t>(setup, layout_.leader_owned_begin, setup->registration_ready_offset, setup->participating_rank_count)[index], kSetupPublication);
        waitForRankPublications(
            moeOverlayExtentArray<std::uint64_t>(setup, layout_.leader_owned_begin, setup->registration_ready_offset, setup->participating_rank_count),
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
        std::uint64_t participant_word_offset = 0u;
        std::vector<std::uint32_t> original_ids(topology_group->participant_ids.begin(), topology_group->participant_ids.end());
        if (!moeOverlaySnapshotMemberOffset(layout_.header, group, original_ids.data(),
                static_cast<std::uint32_t>(member_index), participant_word_offset))
            throw std::logic_error("Device controller snapshot member extent differs from its original group");

        auto *const base = static_cast<std::byte *>(
            mapped_region_->deviceAlias(participant.device));
        MoEOverlayDeviceControllerInferenceEpochRecord epoch_geometry;
        epoch_geometry.bindMemberExtent(group.participant_count);
        const auto epoch_member_index = static_cast<std::size_t>(std::count_if(
            topology_group->participant_ids.begin(), topology_group->participant_ids.end(),
            [participant_id](int id) { return id < participant_id; }));
        MoEOverlayDeviceControllerParticipantBinding binding{
            .device = participant.device,
            .participant_id = participant_id,
            .group_id = topology_group->group_id,
            .capture_metadata = {
                .topology_fingerprint = layout_.header.topology_fingerprint,
                .group_collected_state_words = group.collected_state_words,
                .role_flags = participantFlags(
                    *config_.topology, participant_id, *topology_group),
            },
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
            .logical_targets = at<MoEOverlayDeviceControllerLogicalTargetMetadata>(
                base, static_cast<std::size_t>(layout_.header.logical_target_metadata_offset)),
            .controller = at<MoEOverlayDeviceControllerSharedHeader>(
                base,
                static_cast<std::size_t>(
                    layout_.header.controller_header_offset)),
            .inference_epoch_record = at<
                MoEOverlayDeviceControllerInferenceEpochRecord>(
                base,
                static_cast<std::size_t>(
                    layout_.header.inference_epoch_record_offset)),
            .inference_epoch_storage_bytes = layout_.header.inference_epoch_storage_bytes,
            .inference_epoch_arrival = topology_group->group_id == config_.topology->leader_group_id
                ? at<std::uint64_t>(base, static_cast<std::size_t>(layout_.header.inference_epoch_record_offset +
                    epoch_geometry.arrival_sequences_offset)) + epoch_member_index : nullptr,
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
            .initial_route_targets = at<MoEOverlayRouteTarget>(
                base,
                static_cast<std::size_t>(
                    layout_.header.initial_route_targets_offset)),
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
            .participant_collected_state = at<std::uint64_t>(base, checkedAdd(
                static_cast<std::size_t>(group.collected_state_offset), checkedMultiply(
                    static_cast<std::size_t>(participant_word_offset), sizeof(std::uint64_t),
                    "live participant snapshot bytes"), "live participant snapshot address")),
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

        const auto *ids = record->memberIds(layout_.header.inference_epoch_storage_bytes);
        auto *arrivals = record->arrivals(layout_.header.inference_epoch_storage_bytes);
        description
            << "controller_epoch_barrier{controller_state="
            << loadAcquire(controller->state)
            << ",durable="
            << loadAcquire(controller->current_durable_epoch)
            << ",admission=" << loadAcquire(controller->admission_epoch)
            << ",transaction=" << loadAcquire(controller->transaction_id)
            << ",magic=" << loadAcquire(record->magic)
            << ",version=" << loadAcquire(record->version)
            << ",members=" << record->member_count
            << ",publisher="
            << loadAcquire(record->publisher_participant_id)
            << ",frozen_epoch=" << loadAcquire(record->epoch)
            << ",published_sequence="
            << loadAcquire(record->publication_sequence)
            << ",arrivals=[";
        bool first = true;
        for (std::uint32_t member = 0u; ids && arrivals && member < record->member_count; ++member)
        {
            if (!first) description << ',';
            first = false;
            description << ids[member] << ':' << loadAcquire(arrivals[member]);
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
            economy->logical_target_count !=
                layout_.header.logical_target_count ||
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
        if (profiles.service->target_costs.size() !=
            static_cast<std::size_t>(layout_.header.logical_target_count) *
                layout_.header.num_layers ||
            !profiles.service->production_topology.valid() ||
            profiles.service->production_topology.layerCount() !=
                layout_.header.num_layers ||
            profiles.service->production_topology.firstModelLayer() !=
                layout_.header.first_model_layer)
        {
            throw std::invalid_argument(
                "certified device economy service profile has incomplete logical-target/layer topology");
        }
        const auto active_sources =
            profiles.service->production_topology.economyActiveSources();
        std::uint32_t active_source_bits = 0u;
        for (std::uint32_t phase = 0u;
             phase <
                 kMoEOverlayDeviceControllerEconomyServicePhaseCount;
             ++phase)
        {
            if (active_sources[phase])
                active_source_bits |= 1u << phase;
        }
        if (active_source_bits == 0u)
        {
            throw std::invalid_argument(
                "certified device economy service profile enables no production phase");
        }
        const auto target_index = [this](MoEOverlayRouteTarget target) -> std::size_t
        {
            for (std::size_t index = 0; index < layout_.logical_targets.size(); ++index)
                if (layout_.logical_targets[index].target.word() == target.word()) return index;
            throw std::invalid_argument("Certified economy names a foreign original logical target");
        };
        for (const auto &row : profiles.service->target_costs)
        {
            const auto vertex = target_index(row.target);
            if (row.layer < 0 || !layout_.containsModelLayer(row.layer))
            {
                throw std::invalid_argument(
                    "certified device economy service profile contains an invalid coordinate");
            }
            if (row.target.isCohort())
            {
                const auto &plan = profiles.service->prepared_plan;
                if (!plan || !plan->authenticates(plan))
                    throw std::invalid_argument("Cohort economy requires its original prepared-target service owner");
                const auto &declaration = plan->entry({row.target, row.layer});
                const auto measured = plan->cohort(declaration);
                const auto original = config_.topology->cohortForTarget(row.target, row.layer);
                if (!measured || measured.get() != original.get() || measured.owner_before(original) || original.owner_before(measured))
                    throw std::invalid_argument("Cohort economy changed the original source/cut authority");
            }
            for (std::uint32_t phase = 0u;
                 phase <
                     kMoEOverlayDeviceControllerEconomyServicePhaseCount;
                 ++phase)
            {
                const std::size_t offset =
                    (vertex *
                         layout_.header.num_layers +
                     layout_.storageIndexForModelLayer(row.layer)) *
                        kMoEOverlayDeviceControllerEconomyServicePhaseCount +
                    phase;
                const bool active =
                    profiles.service->production_topology
                        .requiresServiceEvidence(row.layer, phase);
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
        const std::size_t target_count =
            layout_.header.logical_target_count;
        const std::size_t layer_count = layout_.header.num_layers;
        std::vector<bool> migration_seen(
            static_cast<std::size_t>(
                layout_.header.economy_migration_cost_entries),
            false);
        for (std::size_t participant = 0u;
             participant < target_count;
             ++participant)
        {
            for (std::size_t layer = 0u; layer < layer_count; ++layer)
            {
                migration_seen[
                    (participant * target_count + participant) *
                        layer_count +
                    layer] = true;
            }
        }
        if (profiles.migration->costs.size() !=
            target_count * (target_count - 1u) * layer_count)
        {
            throw std::invalid_argument(
                "certified device economy migration profile has incomplete directed geometry");
        }
        for (const auto &row : profiles.migration->costs)
        {
            const auto source = target_index(row.coordinate.source_target);
            const auto destination = target_index(row.coordinate.destination_target);
            if (!row.coordinate.valid() || !layout_.containsModelLayer(row.coordinate.layer) ||
                source == destination || row.transfer_and_repack_ns == 0u)
                throw std::invalid_argument("Certified device migration price has a foreign target/layer or zero transfer price");
            const std::size_t offset = (source * target_count + destination) * layer_count +
                layout_.storageIndexForModelLayer(row.coordinate.layer);
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
        if (loadAcquire(moeOverlayExtentArray<std::uint64_t>(setup, layout_.leader_owned_begin, setup->demand_histogram_rebase_ready_offset, setup->participating_rank_count)[index]) != 0u)
        {
            throw std::logic_error(
                "device controller histogram rebase may be published exactly once per rank");
        }

        // The device terminal events were acquired before this release store.
        // Peers therefore cannot observe the acknowledgement before both
        // process-local phase baselines are durable on every local GPU.
        storeRelease(
            moeOverlayExtentArray<std::uint64_t>(setup, layout_.leader_owned_begin, setup->demand_histogram_rebase_ready_offset, setup->participating_rank_count)[index],
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
                    moeOverlayExtentArray<std::uint64_t>(setup, layout_.leader_owned_begin, setup->demand_histogram_rebase_ready_offset, setup->participating_rank_count)[index]) !=
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
                layout_.header.first_model_layer,
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
               group_records.size() == group_record_count &&
               topology_participant_records.size() == group_record_count &&
               topology_participant_record_counts.size() == group_record_count &&
               participant_records && participant_record_count > 0u &&
               transport && lifetime &&
               layout->magic == kMoEOverlayDeviceControllerFabricMagic &&
               layout->version == kMoEOverlayDeviceControllerFabricVersion &&
               layout->topology_fingerprint == topology_fingerprint &&
               layout->command_capacity == command_capacity &&
               layout->first_model_layer >= 0 && layout->num_layers > 0u &&
               layout->num_layers <= static_cast<std::uint32_t>(
                   std::numeric_limits<std::int32_t>::max() - layout->first_model_layer) &&
               layout->num_experts > 0u && layout->participant_count > 0u &&
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
            const auto target_count =
                topology_participant_record_counts[index];
            if (!record || record->magic != kMoEOverlayDeviceControllerMagic ||
                record->version != kMoEOverlayDeviceControllerVersion ||
                record->group_id != index ||
                record->topology_fingerprint != topology_fingerprint ||
                !participants || target_count == 0u)
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
        binding.group_records.resize(binding.group_record_count);
        binding.topology_participant_records.resize(binding.group_record_count);
        binding.topology_participant_record_counts.resize(binding.group_record_count);
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
