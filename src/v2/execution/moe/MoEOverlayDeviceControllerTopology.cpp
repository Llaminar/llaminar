/**
 * @file MoEOverlayDeviceControllerTopology.cpp
 * @brief Resolution and validation of all-GPU ExpertOverlay controller groups.
 *
 * This file performs setup-only topology work. It never reads a live routing
 * histogram, chooses an expert movement, mirrors device policy on the host, or
 * participates in inference. The resulting value is immutable graph identity
 * consumed later by the mapped-control fabric and participant-local builders.
 */

#include "MoEOverlayDeviceControllerTopology.h"

#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace llaminar2
{
    namespace
    {
        /** Mix trivially represented bytes into one deterministic FNV-1a digest. */
        void mixBytes(
            std::uint64_t &hash,
            const void *data,
            std::size_t bytes) noexcept
        {
            const auto *cursor = static_cast<const unsigned char *>(data);
            for (std::size_t index = 0; index < bytes; ++index)
            {
                hash ^= cursor[index];
                hash *= 1099511628211ULL;
            }
        }

        /** Mix one fixed-width scalar without relying on host object padding. */
        template <typename T>
        void mixScalar(std::uint64_t &hash, T value) noexcept
        {
            mixBytes(hash, &value, sizeof(value));
        }

        /** Mix one string including its length so concatenations are unambiguous. */
        void mixString(std::uint64_t &hash, std::string_view value) noexcept
        {
            mixScalar(hash, static_cast<std::uint64_t>(value.size()));
            mixBytes(hash, value.data(), value.size());
        }

        /** Resolve one declarative domain and preserve its stable ordinal. */
        std::pair<const RoutedExpertDomain *, int> requireDomain(
            const MoERoutedExpertPlacementPlan &plan,
            const std::string &name)
        {
            for (std::size_t index = 0; index < plan.domains.size(); ++index)
            {
                if (plan.domains[index].name == name)
                {
                    return {&plan.domains[index], static_cast<int>(index)};
                }
            }
            throw std::invalid_argument(
                "Device-resident ExpertOverlay controller cannot resolve domain '" +
                name + "'");
        }

        /** Return a physical node id, rejecting every implicit locality guess. */
        int requireNodeId(
            int world_rank,
            std::span<const int> world_rank_node_ids)
        {
            if (world_rank < 0 ||
                static_cast<std::size_t>(world_rank) >=
                    world_rank_node_ids.size())
            {
                throw std::invalid_argument(
                    "Device-resident ExpertOverlay controller requires a physical node id for world rank " +
                    std::to_string(world_rank));
            }
            const int node_id =
                world_rank_node_ids[static_cast<std::size_t>(world_rank)];
            if (node_id < 0)
            {
                throw std::invalid_argument(
                    "Device-resident ExpertOverlay controller received an unresolved physical node id for world rank " +
                    std::to_string(world_rank));
            }
            return node_id;
        }

        /**
         * Return whether a tier-domain can mirror its root with one native
         * rank-local collective. Cross-rank and mixed-vendor domains are split
         * into singleton controller groups and use mapped inter-group records.
         */
        bool supportsNativeIntraGroup(
            const RoutedExpertDomain &domain,
            const std::vector<const MoEExpertOwnerParticipant *> &members)
        {
            if (members.size() < 2u || !domain.isCollectiveDomain())
                return false;
            const DeviceType type = members.front()->address.device_type;
            const int rank = members.front()->world_rank;
            if (type != DeviceType::CUDA && type != DeviceType::ROCm)
                return false;
            for (const auto *member : members)
            {
                if (!member || !member->world_rank_known ||
                    member->world_rank != rank ||
                    member->address.device_type != type)
                {
                    return false;
                }
            }
            return domain.backend == CollectiveBackendType::AUTO ||
                   (type == DeviceType::CUDA &&
                    domain.backend == CollectiveBackendType::NCCL) ||
                   (type == DeviceType::ROCm &&
                    domain.backend == CollectiveBackendType::RCCL);
        }

        /** Return the continuation-root participant that must own policy. */
        const MoEExpertOwnerParticipant &requireLeader(
            const MoERoutedExpertPlacementPlan &plan,
            const MoEExpertOwnerMap &owner_map)
        {
            const std::string &continuation_domain =
                !plan.continuation_domain_spec.domain.empty()
                    ? plan.continuation_domain_spec.domain
                    : plan.continuation_domain;
            if (continuation_domain.empty())
            {
                throw std::invalid_argument(
                    "Device-resident ExpertOverlay controller requires an explicit continuation domain");
            }
            const int root_ordinal =
                plan.continuation_domain_spec.logical_root_participant;
            if (root_ordinal < 0)
            {
                throw std::invalid_argument(
                    "Device-resident ExpertOverlay controller continuation root must be non-negative");
            }

            const MoEExpertOwnerParticipant *leader = nullptr;
            for (const auto &participant : owner_map.participants())
            {
                if (participant.domain_name == continuation_domain &&
                    participant.domain_participant_index == root_ordinal)
                {
                    if (leader)
                    {
                        throw std::invalid_argument(
                            "Device-resident ExpertOverlay controller continuation root is ambiguous across routed tiers");
                    }
                    leader = &participant;
                }
            }
            if (!leader)
            {
                throw std::invalid_argument(
                    "Device-resident ExpertOverlay controller cannot resolve continuation root ordinal " +
                    std::to_string(root_ordinal) + " in domain '" +
                    continuation_domain + "'");
            }
            return *leader;
        }

    } // namespace

    bool MoEOverlayDeviceControllerGroup::valid() const noexcept
    {
        if (group_id < 0 || tier_index < 0 || domain_ordinal < 0 ||
            domain_name.empty() || root_participant_id < 0 ||
            root_world_rank < 0 || !root_device.is_gpu() ||
            participant_ids.empty() ||
            !std::is_sorted(participant_ids.begin(), participant_ids.end()) ||
            !contains(root_participant_id))
        {
            return false;
        }
        return intra_group_transport ==
                   MoEOverlayDeviceControllerIntraGroupTransport::SingleParticipant
                   ? participant_ids.size() == 1u
                   : participant_ids.size() > 1u;
    }

    bool MoEOverlayDeviceControllerGroup::contains(
        int participant_id) const noexcept
    {
        return std::binary_search(
            participant_ids.begin(), participant_ids.end(), participant_id);
    }

    bool MoEOverlayDeviceControllerTopology::valid() const noexcept
    {
        if (leader_participant_id < 0 || leader_group_id < 0 ||
            leader_world_rank < 0 || !leader_device.is_gpu() ||
            participants.size() < 2u || groups.empty() ||
            topology_fingerprint == 0u)
        {
            return false;
        }
        std::vector<bool> covered(participants.size(), false);
        bool leader_group_seen = false;
        for (std::size_t index = 0; index < participants.size(); ++index)
        {
            if (participants[index].participant_id !=
                    static_cast<int>(index) ||
                !participants[index].device.is_gpu())
            {
                return false;
            }
        }
        for (std::size_t index = 0; index < groups.size(); ++index)
        {
            const auto &group = groups[index];
            if (!group.valid() || group.group_id != static_cast<int>(index))
                return false;
            if (group.group_id == leader_group_id)
            {
                leader_group_seen =
                    group.contains(leader_participant_id) &&
                    group.inter_group_transport ==
                        MoEOverlayDeviceControllerInterGroupTransport::LeaderLocal;
            }
            else if (group.inter_group_transport !=
                     MoEOverlayDeviceControllerInterGroupTransport::NodeLocalMapped)
            {
                return false;
            }
            for (const int participant_id : group.participant_ids)
            {
                if (participant_id < 0 ||
                    static_cast<std::size_t>(participant_id) >= covered.size() ||
                    covered[static_cast<std::size_t>(participant_id)])
                {
                    return false;
                }
                covered[static_cast<std::size_t>(participant_id)] = true;
            }
        }
        return leader_group_seen &&
               std::all_of(covered.begin(), covered.end(), [](bool value)
                           { return value; });
    }

    const MoEOverlayDeviceControllerGroup *
    MoEOverlayDeviceControllerTopology::groupForParticipant(
        int participant_id) const noexcept
    {
        const auto found = std::find_if(
            groups.begin(), groups.end(),
            [participant_id](const auto &group)
            {
                return group.contains(participant_id);
            });
        return found == groups.end() ? nullptr : &*found;
    }

    bool MoEOverlayDeviceControllerTopology::usesNodeLocalMappedControl()
        const noexcept
    {
        return std::any_of(
            groups.begin(), groups.end(), [](const auto &group)
            {
                return group.inter_group_transport ==
                       MoEOverlayDeviceControllerInterGroupTransport::NodeLocalMapped;
            });
    }

    MoEOverlayDeviceControllerTopology
    resolveMoEOverlayDeviceControllerTopology(
        const MoERoutedExpertPlacementPlan &plan,
        const MoEExpertOwnerMap &owner_map,
        const MoEOverlayDeviceControllerTopologyOptions &options)
    {
        if (!plan.usesExpertOverlayAuthority() ||
            plan.authority_execution !=
                MoEOverlayAuthorityExecutionKind::DeviceResident)
        {
            throw std::invalid_argument(
                "Device controller topology requires a frozen device-resident ExpertOverlay plan");
        }
        if (owner_map.participants().size() < 2u)
        {
            throw std::invalid_argument(
                "Device controller topology requires at least two routed-expert participants");
        }

        MoEOverlayDeviceControllerTopology topology;
        topology.participants = owner_map.participants();
        const auto &leader = requireLeader(plan, owner_map);
        if (!leader.device.is_gpu() || !leader.world_rank_known)
        {
            throw std::invalid_argument(
                "Device controller leader must be an exact world-rank GPU participant");
        }
        topology.leader_participant_id = leader.participant_id;
        topology.leader_world_rank = leader.world_rank;
        topology.leader_device = leader.device;
        const int leader_node = requireNodeId(
            leader.world_rank, options.world_rank_node_ids);

        for (std::size_t index = 0; index < topology.participants.size(); ++index)
        {
            const auto &participant = topology.participants[index];
            if (participant.participant_id != static_cast<int>(index) ||
                !participant.device.is_gpu() || !participant.world_rank_known)
            {
                throw std::invalid_argument(
                    "Device controller participant catalogue must be dense, all-GPU, and world-rank resolved");
            }
            if (requireNodeId(
                    participant.world_rank,
                    options.world_rank_node_ids) != leader_node)
            {
                throw std::logic_error(
                    "Device-resident ExpertOverlay controller is node-local only; participant " +
                    std::to_string(participant.participant_id) +
                    " is on another physical node");
            }
        }

        for (std::size_t tier_index = 0;
             tier_index < plan.routed_tiers.size(); ++tier_index)
        {
            const auto &tier = plan.routed_tiers[tier_index];
            const auto [domain, domain_ordinal] =
                requireDomain(plan, tier.domain);
            std::vector<const MoEExpertOwnerParticipant *> members;
            for (const auto &participant : topology.participants)
            {
                if (participant.tier_idx == static_cast<int>(tier_index))
                    members.push_back(&participant);
            }
            if (members.empty())
            {
                throw std::invalid_argument(
                    "Device controller tier '" + tier.name +
                    "' has no owner-map participants");
            }

            const bool native_group =
                supportsNativeIntraGroup(*domain, members);
            if (native_group)
            {
                std::vector<int> ids;
                ids.reserve(members.size());
                for (const auto *member : members)
                    ids.push_back(member->participant_id);

                const int root =
                    std::find(ids.begin(), ids.end(), leader.participant_id) !=
                            ids.end()
                        ? leader.participant_id
                        : *std::min_element(ids.begin(), ids.end());
                const auto *root_participant = owner_map.participantForId(root);
                if (!root_participant)
                    throw std::logic_error("Device controller group root is absent");
                topology.groups.push_back({
                    .tier_index = static_cast<int>(tier_index),
                    .tier_priority = tier.priority,
                    .domain_ordinal = domain_ordinal,
                    .domain_name = domain->name,
                    .root_participant_id = root,
                    .root_world_rank = root_participant->world_rank,
                    .root_device = root_participant->device,
                    .participant_ids = std::move(ids),
                    .intra_group_transport =
                        MoEOverlayDeviceControllerIntraGroupTransport::NativeCollective,
                });
            }
            else
            {
                for (const auto *member : members)
                {
                    topology.groups.push_back({
                        .tier_index = static_cast<int>(tier_index),
                        .tier_priority = tier.priority,
                        .domain_ordinal = domain_ordinal,
                        .domain_name = domain->name,
                        .root_participant_id = member->participant_id,
                        .root_world_rank = member->world_rank,
                        .root_device = member->device,
                        .participant_ids = {member->participant_id},
                        .intra_group_transport =
                            MoEOverlayDeviceControllerIntraGroupTransport::SingleParticipant,
                    });
                }
            }
        }

        std::sort(
            topology.groups.begin(), topology.groups.end(),
            [](const auto &left, const auto &right)
            {
                return left.root_participant_id < right.root_participant_id;
            });
        for (std::size_t index = 0; index < topology.groups.size(); ++index)
        {
            auto &group = topology.groups[index];
            group.group_id = static_cast<int>(index);
            if (group.contains(leader.participant_id))
            {
                if (topology.leader_group_id >= 0)
                {
                    throw std::logic_error(
                        "Device controller leader belongs to multiple control groups");
                }
                topology.leader_group_id = group.group_id;
                group.inter_group_transport =
                    MoEOverlayDeviceControllerInterGroupTransport::LeaderLocal;
            }
            else
            {
                group.inter_group_transport =
                    MoEOverlayDeviceControllerInterGroupTransport::NodeLocalMapped;
            }
        }

        std::uint64_t fingerprint = 1469598103934665603ULL;
        mixScalar(fingerprint, topology.leader_participant_id);
        mixScalar(fingerprint, topology.leader_world_rank);
        for (const auto &participant : topology.participants)
        {
            mixScalar(fingerprint, participant.participant_id);
            mixScalar(fingerprint, participant.tier_idx);
            mixScalar(fingerprint, participant.domain_participant_index);
            mixScalar(fingerprint, participant.world_rank);
            mixScalar(
                fingerprint,
                static_cast<std::uint8_t>(participant.address.device_type));
            mixScalar(fingerprint, participant.address.device_ordinal);
            mixString(fingerprint, participant.domain_name);
        }
        for (const auto &group : topology.groups)
        {
            mixScalar(fingerprint, group.group_id);
            mixScalar(fingerprint, group.tier_index);
            mixScalar(fingerprint, group.tier_priority);
            mixScalar(fingerprint, group.domain_ordinal);
            mixScalar(fingerprint, group.root_participant_id);
            mixScalar(fingerprint, group.root_world_rank);
            mixScalar(
                fingerprint,
                static_cast<std::uint8_t>(group.intra_group_transport));
            mixScalar(
                fingerprint,
                static_cast<std::uint8_t>(group.inter_group_transport));
            mixString(fingerprint, group.domain_name);
            for (const int participant_id : group.participant_ids)
                mixScalar(fingerprint, participant_id);
        }
        topology.topology_fingerprint = fingerprint == 0u ? 1u : fingerprint;
        if (!topology.valid())
        {
            throw std::logic_error(
                "Resolved device-resident ExpertOverlay controller topology is internally inconsistent");
        }
        return topology;
    }

    const char *toString(
        MoEOverlayDeviceControllerIntraGroupTransport transport) noexcept
    {
        switch (transport)
        {
        case MoEOverlayDeviceControllerIntraGroupTransport::SingleParticipant:
            return "single_participant";
        case MoEOverlayDeviceControllerIntraGroupTransport::NativeCollective:
            return "native_collective";
        }
        return "invalid";
    }

    const char *toString(
        MoEOverlayDeviceControllerInterGroupTransport transport) noexcept
    {
        switch (transport)
        {
        case MoEOverlayDeviceControllerInterGroupTransport::LeaderLocal:
            return "leader_local";
        case MoEOverlayDeviceControllerInterGroupTransport::NodeLocalMapped:
            return "node_local_mapped";
        }
        return "invalid";
    }
} // namespace llaminar2
