/**
 * @file MoEOverlayActivationChannelPlan.cpp
 * @brief Shared node-local ExpertOverlay activation topology/BOM implementation.
 *
 * This file performs no allocation and has no MPI dependency. It translates
 * the already hardware-bound placement into the exact lane set later consumed
 * by transport preflight, allowing automatic expert capacity to reserve those
 * same capture-stable buffers before any model weight is placed.
 */

#include "MoEOverlayActivationChannelPlan.h"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

namespace llaminar2
{
    namespace
    {
        /** @brief Stable aggregate key for one physical allocation authority. */
        struct PhysicalKey
        {
            int world_rank = -1;
            DeviceId device = DeviceId::invalid();

            /** @return Strict ordering for deterministic maps and diagnostics. */
            [[nodiscard]] bool operator<(
                const PhysicalKey &other) const noexcept
            {
                if (world_rank != other.world_rank)
                    return world_rank < other.world_rank;
                return device < other.device;
            }
        };

        /** @brief Checked product used by every matrix and family charge. */
        [[nodiscard]] std::size_t checkedMultiply(
            std::size_t left,
            std::size_t right,
            const char *what)
        {
            if (left != 0u &&
                right > std::numeric_limits<std::size_t>::max() / left)
            {
                throw std::overflow_error(
                    std::string("ExpertOverlay activation channel ") + what +
                    " overflows size_t");
            }
            return left * right;
        }

        /** @brief Checked sum used while coalescing lanes onto one device. */
        [[nodiscard]] std::size_t checkedAdd(
            std::size_t left,
            std::size_t right,
            const char *what)
        {
            if (right > std::numeric_limits<std::size_t>::max() - left)
            {
                throw std::overflow_error(
                    std::string("ExpertOverlay activation channel ") + what +
                    " overflows size_t");
            }
            return left + right;
        }

        /**
         * @brief Index the gathered topology by its explicit world-rank field.
         *
         * Vector position is intentionally not treated as authority. Requiring
         * total unique rank identities catches a stale/incomplete gather before
         * admission and runtime preflight can disagree about node locality.
         */
        [[nodiscard]] std::map<int, const RankInventory *> rankIndex(
            const ClusterInventory &inventory)
        {
            if (inventory.world_size <= 0 ||
                inventory.ranks.size() !=
                    static_cast<std::size_t>(inventory.world_size))
            {
                throw std::invalid_argument(
                    "ExpertOverlay activation planning requires a complete cluster inventory");
            }
            std::map<int, const RankInventory *> result;
            for (const auto &rank : inventory.ranks)
            {
                if (rank.rank < 0 || rank.rank >= inventory.world_size ||
                    rank.node_id < 0 ||
                    !result.emplace(rank.rank, &rank).second)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay activation planning requires unique rank and node identities");
                }
            }
            return result;
        }

        /** @brief Resolve a tier's exact routed-domain declaration ordinal. */
        [[nodiscard]] int domainOrdinal(
            const MoERoutedExpertPlacementPlan &plan,
            int tier_index)
        {
            if (tier_index < 0 ||
                static_cast<std::size_t>(tier_index) >=
                    plan.routed_tiers.size())
            {
                throw std::invalid_argument(
                    "ExpertOverlay activation participant references an invalid tier");
            }
            const std::string &domain_name =
                plan.routed_tiers[static_cast<std::size_t>(tier_index)].domain;
            for (std::size_t index = 0; index < plan.domains.size(); ++index)
            {
                if (plan.domains[index].name == domain_name)
                    return static_cast<int>(index);
            }
            throw std::invalid_argument(
                "ExpertOverlay activation tier references an unknown domain '" +
                domain_name + "'");
        }
    } // namespace

    std::vector<int>
    MoEOverlayNodeLocalActivationChannelPlan::targetParticipantIds() const
    {
        std::vector<int> result;
        result.reserve(target_lanes.size());
        for (const auto &lane : target_lanes)
            result.push_back(lane.participant_id);
        return result;
    }

    const std::vector<MoEOverlayActivationLocalLaneBinding> &
    MoEOverlayNodeLocalActivationChannelPlan::localLanes(
        int world_rank) const
    {
        if (world_rank == source_world_rank)
            return source_lanes;
        if (world_rank == target_world_rank)
            return target_lanes;
        throw std::invalid_argument(
            "ExpertOverlay activation lane requested by a non-endpoint rank");
    }

    std::size_t MoEOverlayActivationChannelPlan::stagingBytesFor(
        int world_rank,
        DeviceId device) const noexcept
    {
        const auto found = std::find_if(
            staging_charges.begin(), staging_charges.end(),
            [&](const auto &charge)
            {
                return charge.world_rank == world_rank &&
                       charge.device == device;
            });
        return found == staging_charges.end() ? 0u : found->bytes;
    }

    bool MoEOverlayActivationChannelPlanner::hasRemoteRankParticipants(
        const MoERoutedExpertPlacementPlan &placement)
    {
        const auto participants =
            MoEOverlayCapacityAdmission::boundParticipants(placement);
        const int source_participant_id =
            placement.continuation_domain_spec.logical_root_participant;
        const auto source = std::find_if(
            participants.begin(), participants.end(),
            [&](const auto &participant)
            { return participant.participant_id == source_participant_id; });
        if (source == participants.end() || source->world_rank < 0)
        {
            throw std::invalid_argument(
                "ExpertOverlay remote-rank admission requires a bound continuation root participant");
        }
        return std::any_of(
            participants.begin(), participants.end(),
            [&](const auto &participant)
            {
                return participant.participant_id != source_participant_id &&
                       participant.world_rank != source->world_rank;
            });
    }

    MoEOverlayActivationChannelPlan
    MoEOverlayActivationChannelPlanner::plan(
        const MoEOverlayActivationChannelPlannerInput &input)
    {
        if (!input.placement_plan || !input.cluster_inventory ||
            input.row_capacity == 0u || input.d_model <= 0 ||
            input.top_k <= 0 ||
            input.graph_family_count == 0u)
        {
            throw std::invalid_argument(
                "ExpertOverlay activation planning requires bound topology and positive graph geometry");
        }

        const auto &placement = *input.placement_plan;
        const auto rank_by_id = rankIndex(*input.cluster_inventory);
        const auto participants =
            MoEOverlayCapacityAdmission::boundParticipants(placement);
        const int source_participant_id =
            placement.continuation_domain_spec.logical_root_participant;
        const auto source = std::find_if(
            participants.begin(), participants.end(),
            [&](const auto &participant)
            { return participant.participant_id == source_participant_id; });
        if (source == participants.end() || source->world_rank < 0 ||
            !source->device.is_valid() ||
            (!source->device.is_cpu() && !source->device.is_gpu()))
        {
            throw std::invalid_argument(
                "ExpertOverlay activation planning requires a supported continuation root participant");
        }
        const auto source_rank = rank_by_id.find(source->world_rank);
        if (source_rank == rank_by_id.end())
        {
            throw std::invalid_argument(
                "ExpertOverlay continuation root is absent from cluster inventory");
        }

        const std::size_t row_elements = checkedMultiply(
            input.row_capacity,
            static_cast<std::size_t>(input.d_model),
            "payload elements");
        const std::size_t matrix_bytes = checkedMultiply(
            row_elements, sizeof(float), "payload matrix bytes");
        const std::size_t canonical_route_elements = checkedMultiply(
            row_elements,
            static_cast<std::size_t>(input.top_k),
            "canonical route elements");
        const std::size_t canonical_route_matrix_bytes = checkedMultiply(
            canonical_route_elements,
            sizeof(float),
            "canonical route matrix bytes");
        const std::size_t lane_bytes = checkedMultiply(
            matrix_bytes,
            input.graph_family_count,
            "retained-family lane bytes");

        using GroupKey = std::tuple<int, int, int>;
        std::map<GroupKey, std::vector<MoEOverlayBoundTierParticipant>> groups;
        for (const auto &participant : participants)
        {
            if (participant.participant_id == source_participant_id)
            {
                continue;
            }
            if (participant.world_rank == source->world_rank)
            {
                /*
                 * A rank may own the continuation device and a participant in
                 * another tier (for example one socket of a NodeTP CPU tier).
                 * That participant is already part of the rank-local graph and
                 * its compact activation arena is priced by the ordinary local
                 * participant BOM. This planner owns only cross-rank channels;
                 * manufacturing a self-channel would violate the transport's
                 * two-rank identity and double-charge those local buffers.
                 */
                continue;
            }
            const auto target_rank = rank_by_id.find(participant.world_rank);
            if (target_rank == rank_by_id.end())
            {
                throw std::invalid_argument(
                    "ExpertOverlay activation participant is absent from cluster inventory");
            }
            if (target_rank->second->node_id != source_rank->second->node_id)
            {
                // A different physical node uses the MPI activation transport.
                continue;
            }
            const int domain_ordinal =
                domainOrdinal(placement, participant.tier_index);
            groups[GroupKey{
                participant.tier_index,
                domain_ordinal,
                participant.world_rank}]
                .push_back(participant);
        }

        MoEOverlayActivationChannelPlan result;
        result.row_capacity = input.row_capacity;
        result.d_model = input.d_model;
        result.top_k = input.top_k;
        result.graph_family_count = input.graph_family_count;
        result.payload_matrix_bytes = matrix_bytes;
        result.canonical_route_matrix_bytes =
            canonical_route_matrix_bytes;
        std::map<PhysicalKey, std::size_t> charge_by_resource;
        for (auto &[key, grouped] : groups)
        {
            std::sort(
                grouped.begin(), grouped.end(),
                [](const auto &left, const auto &right)
                { return left.participant_id < right.participant_id; });

            const auto [tier_index, domain_ordinal, target_world_rank] = key;
            MoEOverlayNodeLocalActivationChannelPlan channel;
            channel.tier_index = tier_index;
            channel.domain_ordinal = domain_ordinal;
            channel.source_world_rank = source->world_rank;
            channel.target_world_rank = target_world_rank;
            channel.source_device = source->device;
            channel.source_lanes.reserve(grouped.size());
            channel.target_lanes.reserve(grouped.size());
            for (const auto &participant : grouped)
            {
                const MoEOverlayActivationLocalLaneBinding source_lane{
                    .participant_id = participant.participant_id,
                    .device = source->device,
                };
                const MoEOverlayActivationLocalLaneBinding target_lane{
                    .participant_id = participant.participant_id,
                    .device = participant.device,
                };
                if (!source_lane.valid() || !target_lane.valid())
                {
                    throw std::invalid_argument(
                        "ExpertOverlay node-local activation channels require supported CPU/GPU endpoints");
                }
                channel.source_lanes.push_back(source_lane);
                channel.target_lanes.push_back(target_lane);
                if (source->device.is_gpu())
                {
                    auto &bytes = charge_by_resource[PhysicalKey{
                        source->world_rank, source->device}];
                    bytes = checkedAdd(
                        bytes, lane_bytes, "per-device staging bytes");
                }
                if (participant.device.is_gpu())
                {
                    auto &bytes = charge_by_resource[PhysicalKey{
                        participant.world_rank, participant.device}];
                    bytes = checkedAdd(
                        bytes, lane_bytes, "per-device staging bytes");
                    /* The follower's route bank is participant-owned and
                     * serially shared by main, decode, verifier, and every MTP
                     * family. Charge it once per logical participant—not once
                     * per captured graph family—at the planner's maximum row
                     * geometry. */
                    bytes = checkedAdd(
                        bytes,
                        canonical_route_matrix_bytes,
                        "participant canonical route bytes");
                }
            }
            result.channels.push_back(std::move(channel));
        }

        result.staging_charges.reserve(charge_by_resource.size());
        for (const auto &[resource, bytes] : charge_by_resource)
        {
            result.staging_charges.push_back({
                .world_rank = resource.world_rank,
                .device = resource.device,
                .bytes = bytes,
            });
        }
        return result;
    }
} // namespace llaminar2
