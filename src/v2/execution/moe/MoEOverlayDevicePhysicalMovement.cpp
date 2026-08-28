/**
 * @file MoEOverlayDevicePhysicalMovement.cpp
 * @brief Validation and endpoint resolution for device-authored MoE movement.
 */

#include "MoEOverlayDevicePhysicalMovement.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace llaminar2
{
    namespace
    {
        using RequirementKey = std::tuple<int, int, int>;

        /** Mix one canonical word into two independent transaction lanes. */
        void mixWord(
            MoEOverlayResidencyExecutionFingerprint &fingerprint,
            std::uint64_t value,
            std::uint64_t ordinal) noexcept
        {
            std::uint64_t lane = value +
                0x9e3779b97f4a7c15ULL * (ordinal + 1u);
            lane = (lane ^ (lane >> 30u)) * 0xbf58476d1ce4e5b9ULL;
            lane = (lane ^ (lane >> 27u)) * 0x94d049bb133111ebULL;
            lane ^= lane >> 31u;

            fingerprint.low ^= lane;
            fingerprint.low =
                std::rotl(fingerprint.low, 23) * 0x9e3779b185ebca87ULL;

            lane ^= 0xd6e8feb86659fd93ULL;
            lane = (lane ^ (lane >> 32u)) * 0xd6e8feb86659fd93ULL;
            lane = (lane ^ (lane >> 32u)) * 0xa5a3564e27f8862fULL;
            lane ^= lane >> 32u;
            fingerprint.high += lane;
            fingerprint.high =
                std::rotl(fingerprint.high, 37) * 0x94d049bb133111ebULL;
        }

        /** Fingerprint every semantic command field without object padding. */
        MoEOverlayResidencyExecutionFingerprint fingerprintCommand(
            const MoEOverlayDeviceTransportCommandBatch &command) noexcept
        {
            MoEOverlayResidencyExecutionFingerprint fingerprint{
                .low = 0x243f6a8885a308d3ULL,
                .high = 0x13198a2e03707344ULL,
            };
            std::uint64_t ordinal = 0u;
            auto mix = [&](std::uint64_t value)
            {
                mixWord(fingerprint, value, ordinal++);
            };

            mix(command.header.magic);
            mix(command.header.version);
            mix(command.header.kind);
            mix(command.header.command_count);
            mix(command.header.topology_fingerprint);
            mix(command.header.transaction_id);
            mix(command.header.base_epoch);
            mix(command.header.candidate_epoch);
            mix(command.header.command_digest);
            mix(command.header.packed_weight_bytes);
            mix(command.header.parallel_command_count);
            mix(command.header.movement_round_count);
            mix(command.header.hazard_count);
            mix(command.participant_count);
            mix(command.num_layers);
            mix(command.num_experts);
            for (const auto &entry : command.entries)
            {
                mix(entry.magic);
                mix(entry.version);
                mix(entry.op);
                mix(entry.ordinal);
                mix(entry.layer);
                mix(entry.expert);
                mix(entry.source_participant);
                mix(entry.destination_participant);
                mix(entry.payload_slot);
                mix(entry.flags);
                mix(entry.payload_bytes);
                mix(entry.source_epoch);
                mix(entry.candidate_epoch);
            }

            if (fingerprint.low == 0u)
                fingerprint.low = 0x6a09e667f3bcc909ULL;
            if (fingerprint.high == 0u)
                fingerprint.high = 0xbb67ae8584caa73bULL;
            return fingerprint;
        }

        /** Resolve one command participant into a complete physical owner. */
        MoEExpertOwner physicalOwner(
            const MoEOverlayDeviceControllerTopology &topology,
            std::uint32_t participant_id,
            std::uint32_t layer,
            std::uint32_t expert)
        {
            if (participant_id >= topology.participants.size())
                throw std::invalid_argument(
                    "device physical movement names an unknown participant");
            const auto &participant = topology.participants[participant_id];
            if (participant.participant_id !=
                    static_cast<int>(participant_id) ||
                !participant.device.is_gpu())
            {
                throw std::invalid_argument(
                    "device physical movement participant catalogue is not dense all-GPU topology");
            }
            return {
                .layer_idx = static_cast<int>(layer),
                .expert_id = static_cast<int>(expert),
                .tier_idx = participant.tier_idx,
                .owner_participant = participant.participant_id,
                .device = participant.device,
                .resident = true,
                .tier_name = participant.tier_name,
                .domain_name = participant.domain_name,
                .domain_participant_index =
                    participant.domain_participant_index,
                .owner_world_rank = participant.world_rank,
                .owner_world_rank_known = participant.world_rank_known,
                .address = participant.address,
            };
        }

        /** Resolve migration direction strictly from opaque integer priority. */
        MoEOverlayTierMigrationDirection movementDirection(
            const MoEOverlayDeviceControllerTopology &topology,
            int source_participant,
            int destination_participant)
        {
            const auto *source =
                topology.groupForParticipant(source_participant);
            const auto *destination =
                topology.groupForParticipant(destination_participant);
            if (!source || !destination)
                throw std::invalid_argument(
                    "device physical movement cannot resolve participant priority");
            if (destination->tier_priority < source->tier_priority)
                return MoEOverlayTierMigrationDirection::Promotion;
            if (destination->tier_priority > source->tier_priority)
                return MoEOverlayTierMigrationDirection::Demotion;
            return MoEOverlayTierMigrationDirection::SamePriority;
        }

        /**
         * @brief Authenticate the device-authored objective without inference.
         * @param encoded_axis Fixed command word written by the policy leader.
         * @return Shared authority-ledger spelling of the same objective.
         * @throws std::invalid_argument for a non-total wire value.
         */
        MoEOptimizationMovementAxis movementAxis(
            std::uint32_t encoded_axis)
        {
            switch (static_cast<MoEOverlayDeviceMovementAxis>(encoded_axis))
            {
            case MoEOverlayDeviceMovementAxis::TierResidency:
                return MoEOptimizationMovementAxis::TierResidency;
            case MoEOverlayDeviceMovementAxis::ParticipantPlacement:
                return MoEOptimizationMovementAxis::ParticipantPlacement;
            case MoEOverlayDeviceMovementAxis::Combined:
                return MoEOptimizationMovementAxis::Combined;
            }
            throw std::invalid_argument(
                "device physical movement command has an invalid objective axis");
        }

        /** Build exact destination slot demand from physical arrivals. */
        std::vector<MoEOverlayTierShadowRequirement> shadowRequirements(
            const std::vector<MoEOverlayTierMigration> &migrations)
        {
            std::map<RequirementKey, std::size_t> counts;
            for (const auto &migration : migrations)
            {
                ++counts[{migration.layer_idx,
                          migration.destination.tier_idx,
                          migration.destination.owner_participant}];
            }

            std::vector<MoEOverlayTierShadowRequirement> result;
            result.reserve(counts.size());
            for (const auto &[key, count] : counts)
            {
                const auto [layer, tier, participant] = key;
                result.push_back({
                    .layer_idx = layer,
                    .tier_idx = tier,
                    .destination_participant = participant,
                    .slot_count = count,
                });
            }
            return result;
        }

        /**
         * Decompose a balanced directed participant graph into deterministic
         * Euler circuits.  The command order is destination-canonical, so each
         * adjacency list deliberately retains ascending command ordinal.
         */
        std::vector<MoEOverlayTierMigrationCycle> durableCycles(
            const std::vector<MoEOverlayTierMigration> &migrations,
            std::uint32_t participant_count,
            std::uint32_t num_layers)
        {
            std::vector<MoEOverlayTierMigrationCycle> result;
            std::vector<bool> used(migrations.size(), false);

            for (std::uint32_t layer = 0u; layer < num_layers; ++layer)
            {
                std::vector<std::vector<std::size_t>> outgoing(
                    participant_count);
                std::vector<std::size_t> incoming_count(participant_count, 0u);
                for (std::size_t index = 0u; index < migrations.size(); ++index)
                {
                    const auto &migration = migrations[index];
                    if (migration.layer_idx != static_cast<int>(layer))
                        continue;
                    const auto source = static_cast<std::size_t>(
                        migration.source.owner_participant);
                    const auto destination = static_cast<std::size_t>(
                        migration.destination.owner_participant);
                    if (source >= participant_count ||
                        destination >= participant_count)
                    {
                        throw std::invalid_argument(
                            "device Dynamic movement contains an out-of-range cycle endpoint");
                    }
                    outgoing[source].push_back(index);
                    ++incoming_count[destination];
                }
                for (std::size_t participant = 0u;
                     participant < participant_count;
                     ++participant)
                {
                    if (outgoing[participant].size() !=
                        incoming_count[participant])
                    {
                        throw std::invalid_argument(
                            "device Dynamic movement is not capacity-preserving within a layer");
                    }
                }

                std::vector<std::size_t> cursor(participant_count, 0u);
                for (std::size_t first = 0u; first < migrations.size(); ++first)
                {
                    if (used[first] ||
                        migrations[first].layer_idx !=
                            static_cast<int>(layer))
                    {
                        continue;
                    }

                    const int start =
                        migrations[first].source.owner_participant;
                    std::vector<int> vertex_stack{start};
                    std::vector<std::size_t> edge_stack;
                    std::vector<std::size_t> reverse_circuit;
                    while (!vertex_stack.empty())
                    {
                        const int vertex = vertex_stack.back();
                        if (vertex < 0 ||
                            static_cast<std::size_t>(vertex) >=
                                outgoing.size())
                        {
                            throw std::invalid_argument(
                                "device Dynamic movement cycle lost a participant");
                        }
                        auto &next = cursor[static_cast<std::size_t>(vertex)];
                        auto &edges = outgoing[static_cast<std::size_t>(vertex)];
                        while (next < edges.size() && used[edges[next]])
                            ++next;
                        if (next < edges.size())
                        {
                            const std::size_t edge = edges[next++];
                            used[edge] = true;
                            edge_stack.push_back(edge);
                            vertex_stack.push_back(
                                migrations[edge]
                                    .destination.owner_participant);
                            continue;
                        }

                        vertex_stack.pop_back();
                        if (!edge_stack.empty())
                        {
                            reverse_circuit.push_back(edge_stack.back());
                            edge_stack.pop_back();
                        }
                    }
                    std::reverse(
                        reverse_circuit.begin(), reverse_circuit.end());
                    MoEOverlayTierMigrationCycle cycle{
                        .layer_idx = static_cast<int>(layer),
                        .migration_indices = std::move(reverse_circuit),
                    };
                    if (!cycle.valid(migrations))
                    {
                        throw std::invalid_argument(
                            "device Dynamic movement could not be decomposed into closed physical cycles");
                    }
                    result.push_back(std::move(cycle));
                }
            }

            if (std::any_of(used.begin(), used.end(), [](bool value)
                            { return !value; }))
            {
                throw std::invalid_argument(
                    "device Dynamic movement cycle decomposition did not cover every command");
            }
            return result;
        }

        /** Compare batch slot demand with the exact migrations it represents. */
        bool shadowRequirementsValid(
            const std::vector<MoEOverlayTierMigration> &migrations,
            const std::vector<MoEOverlayTierShadowRequirement> &requirements)
            noexcept
        {
            std::map<RequirementKey, std::size_t> expected;
            for (const auto &migration : migrations)
            {
                ++expected[{migration.layer_idx,
                            migration.destination.tier_idx,
                            migration.destination.owner_participant}];
            }
            std::map<RequirementKey, std::size_t> actual;
            for (const auto &requirement : requirements)
            {
                if (requirement.layer_idx < 0 || requirement.tier_idx < 0 ||
                    requirement.destination_participant < 0 ||
                    requirement.slot_count == 0u ||
                    !actual.emplace(
                               RequirementKey{
                                   requirement.layer_idx,
                                   requirement.tier_idx,
                                   requirement.destination_participant},
                               requirement.slot_count)
                         .second)
                {
                    return false;
                }
            }
            return actual == expected;
        }
    } // namespace

    MoEOverlayPhysicalWavePollCursor::MoEOverlayPhysicalWavePollCursor(
        std::size_t operation_count,
        std::size_t maximum_polls_per_quantum)
        : operation_count_(operation_count),
          maximum_polls_per_quantum_(maximum_polls_per_quantum)
    {
        if (operation_count_ == 0u || maximum_polls_per_quantum_ == 0u)
        {
            throw std::invalid_argument(
                "physical wave poll cursor requires positive operation and quantum geometry");
        }
    }

    std::optional<std::size_t>
    MoEOverlayPhysicalWavePollCursor::nextPending(
        std::span<const std::uint8_t> ready)
    {
        if (ready.size() != operation_count_)
        {
            throw std::invalid_argument(
                "physical wave poll cursor readiness geometry changed");
        }
        while (inspected_this_quantum_ < operation_count_)
        {
            const std::size_t candidate = next_operation_;
            next_operation_ = (next_operation_ + 1u) % operation_count_;
            ++inspected_this_quantum_;
            if (ready[candidate] == 0u)
                return candidate;
        }
        return std::nullopt;
    }

    bool MoEOverlayDevicePhysicalMovementBatch::valid() const noexcept
    {
        if (kind == MoEOverlayDeviceControllerTransactionKind::Invalid ||
            topology_fingerprint == 0u || transaction_id == 0u ||
            base_epoch == 0u || command_digest == 0u ||
            participant_count < 2u || num_layers == 0u ||
            num_experts == 0u || !execution_fingerprint.valid())
        {
            return false;
        }

        std::uint64_t bytes = 0u;
        std::set<std::pair<int, int>> durable_experts;
        for (const auto &migration : migrations)
        {
            if (migration.layer_idx < 0 || migration.expert_id < 0 ||
                static_cast<std::uint32_t>(migration.layer_idx) >= num_layers ||
                static_cast<std::uint32_t>(migration.expert_id) >= num_experts ||
                migration.source.layer_idx != migration.layer_idx ||
                migration.destination.layer_idx != migration.layer_idx ||
                migration.source.expert_id != migration.expert_id ||
                migration.destination.expert_id != migration.expert_id ||
                migration.source.owner_participant < 0 ||
                migration.destination.owner_participant < 0 ||
                migration.source.owner_participant ==
                    migration.destination.owner_participant ||
                !migration.source.device.is_gpu() ||
                !migration.destination.device.is_gpu() ||
                (kind == MoEOverlayDeviceControllerTransactionKind::
                             DynamicPlacement &&
                 migration.axis ==
                     MoEOptimizationMovementAxis::ParticipantPlacement &&
                 migration.crossesTier()) ||
                migration.estimated_weight_bytes == 0u ||
                migration.estimated_weight_bytes >
                    std::numeric_limits<std::uint64_t>::max() - bytes)
            {
                return false;
            }
            bytes += static_cast<std::uint64_t>(
                migration.estimated_weight_bytes);
            if (kind ==
                    MoEOverlayDeviceControllerTransactionKind::DynamicPlacement &&
                !durable_experts.emplace(
                     migration.layer_idx, migration.expert_id).second)
            {
                return false;
            }
        }
        if (bytes != packed_weight_bytes ||
            !shadowRequirementsValid(migrations, shadow_requirements))
        {
            return false;
        }

        switch (kind)
        {
        case MoEOverlayDeviceControllerTransactionKind::StaticCheck:
            return command_count == 0u && candidate_epoch == base_epoch &&
                   migrations.empty() && migration_cycles.empty() &&
                   shadow_requirements.empty() && packed_weight_bytes == 0u;
        case MoEOverlayDeviceControllerTransactionKind::DynamicPlacement:
        {
            if (migrations.empty())
            {
                return command_count == 0u && candidate_epoch == base_epoch &&
                       migration_cycles.empty() && packed_weight_bytes == 0u;
            }
            if (command_count != migrations.size() ||
                candidate_epoch != base_epoch + 1u)
            {
                return false;
            }
            std::vector<std::size_t> coverage(migrations.size(), 0u);
            for (const auto &cycle : migration_cycles)
            {
                if (!cycle.valid(migrations))
                    return false;
                for (const auto index : cycle.migration_indices)
                {
                    if (index >= coverage.size())
                        return false;
                    ++coverage[index];
                }
            }
            return std::all_of(
                coverage.begin(), coverage.end(), [](std::size_t count)
                { return count == 1u; });
        }
        case MoEOverlayDeviceControllerTransactionKind::CurrentBatchLLEP:
            return command_count != 0u && candidate_epoch == base_epoch &&
                   migration_cycles.empty() &&
                   ((migrations.empty() && packed_weight_bytes == 0u) ||
                    (!migrations.empty() && packed_weight_bytes != 0u));
        case MoEOverlayDeviceControllerTransactionKind::Invalid:
            return false;
        }
        return false;
    }

    MoEOverlayDevicePhysicalMovementBatch
    makeMoEOverlayDevicePhysicalMovementBatch(
        const MoEOverlayDeviceTransportCommandBatch &command,
        const MoEOverlayDeviceControllerTopology &topology)
    {
        if (!command.valid())
            throw std::invalid_argument(
                "device physical movement requires an authenticated command batch");
        if (!topology.valid())
            throw std::invalid_argument(
                "device physical movement requires a valid frozen topology");
        if (command.header.topology_fingerprint !=
                topology.topology_fingerprint ||
            command.participant_count != topology.participants.size())
        {
            throw std::invalid_argument(
                "device physical movement command and topology identity differ");
        }

        MoEOverlayDevicePhysicalMovementBatch result{
            .kind = static_cast<MoEOverlayDeviceControllerTransactionKind>(
                command.header.kind),
            .topology_fingerprint = command.header.topology_fingerprint,
            .transaction_id = command.header.transaction_id,
            .base_epoch = command.header.base_epoch,
            .candidate_epoch = command.header.candidate_epoch,
            .command_digest = command.header.command_digest,
            .packed_weight_bytes = command.header.packed_weight_bytes,
            .command_count = command.header.command_count,
            .participant_count = command.participant_count,
            .num_layers = command.num_layers,
            .num_experts = command.num_experts,
            .execution_fingerprint = fingerprintCommand(command),
        };

        for (const auto &entry : command.entries)
        {
            const auto op =
                static_cast<MoEOverlayDeviceMovementOp>(entry.op);
            if (op == MoEOverlayDeviceMovementOp::TransientAssignment)
                continue;
            if (entry.payload_bytes >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max()))
            {
                throw std::invalid_argument(
                    "device physical movement payload exceeds local size_t");
            }
            auto source = physicalOwner(
                topology,
                entry.source_participant,
                entry.layer,
                entry.expert);
            auto destination = physicalOwner(
                topology,
                entry.destination_participant,
                entry.layer,
                entry.expert);
            result.migrations.push_back({
                .layer_idx = static_cast<int>(entry.layer),
                .expert_id = static_cast<int>(entry.expert),
                .activation_count = 0u,
                .estimated_weight_bytes =
                    static_cast<std::size_t>(entry.payload_bytes),
                .direction = movementDirection(
                    topology,
                    source.owner_participant,
                    destination.owner_participant),
                .axis = movementAxis(entry.flags),
                .source = std::move(source),
                .destination = std::move(destination),
            });
        }

        result.shadow_requirements = shadowRequirements(result.migrations);
        if (result.kind ==
                MoEOverlayDeviceControllerTransactionKind::DynamicPlacement &&
            !result.migrations.empty())
        {
            result.migration_cycles = durableCycles(
                result.migrations,
                result.participant_count,
                result.num_layers);
        }

        if (!result.valid())
            throw std::invalid_argument(
                "device physical movement commands do not form a valid physical wave");
        return result;
    }
} // namespace llaminar2
