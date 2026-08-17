/**
 * @file MoEExpertOwnerMap.cpp
 * @brief Deterministic initial and epoch-transition routed-expert assignment.
 *
 * Logical tier placement is converted to exact physical ownership here. The
 * transition path first retains compatible prior owners and then fills only
 * participant deficits in canonical order. That rule avoids gratuitous
 * same-tier copies while keeping independently planning MPI ranks bit-for-bit
 * deterministic.
 */

#include "MoEExpertOwnerMap.h"
#include "RoutedExpertOwnerAssignment.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        std::string formatValidationErrors(const MoERoutedExpertPlacementValidationResult &validation)
        {
            std::ostringstream message;
            message << "Invalid MoE expert owner map plan:";
            for (const auto &error : validation.errors)
                message << "\n - " << error;
            return message.str();
        }

        const RoutedExpertDomain &requireDomain(
            const MoERoutedExpertPlacementPlan &plan,
            const std::string &domain_name,
            const char *context)
        {
            auto it = std::find_if(plan.domains.begin(), plan.domains.end(), [&](const auto &domain)
                                   { return domain.name == domain_name; });
            if (it == plan.domains.end())
            {
                throw std::invalid_argument(std::string("MoE expert owner map ") + context +
                                            " references unknown domain '" + domain_name + "'");
            }
            return *it;
        }

        int participantWorldRank(const RoutedExpertDomain &domain, size_t participant_index)
        {
            if (participant_index < domain.world_ranks.size())
                return domain.world_ranks[participant_index];
            if (domain.scope == ExecutionDomainScope::NODE_LOCAL)
                return static_cast<int>(participant_index);
            if (domain.owner_rank >= 0)
                return domain.owner_rank;
            return -1;
        }

        bool participantWorldRankKnown(const RoutedExpertDomain &domain, size_t participant_index)
        {
            return participant_index < domain.world_ranks.size() ||
                   domain.scope == ExecutionDomainScope::NODE_LOCAL ||
                   domain.owner_rank >= 0;
        }

        std::vector<MoEExpertOwnerParticipant> buildTierParticipants(
            const MoERoutedExpertPlacementPlan &plan,
            size_t tier_index,
            const MoEExpertOwnerMapBuildOptions &options,
            int first_participant_id)
        {
            const auto &tier = plan.routed_tiers[tier_index];
            const auto &domain = requireDomain(plan, tier.domain, "routed tier");

            if (options.reject_tensor_sharded_domains &&
                domain.routed_compute_policy == RoutedExpertComputePolicy::TensorSharded)
            {
                std::ostringstream message;
                message << "Graph-native routed tier '" << tier.name << "' in domain '" << domain.name
                        << "' uses routed_compute=tensor-sharded; a whole-expert owner map cannot represent within-expert tensor shards";
                throw std::invalid_argument(message.str());
            }

            std::vector<MoEExpertOwnerParticipant> participants;
            participants.reserve(domain.participants.size());
            for (size_t participant_index = 0; participant_index < domain.participants.size(); ++participant_index)
            {
                const auto &address = domain.participants[participant_index];
                MoEExpertOwnerParticipant participant;
                participant.participant_id = first_participant_id + static_cast<int>(participant_index);
                participant.tier_idx = static_cast<int>(tier_index);
                participant.tier_name = tier.name;
                participant.domain_name = tier.domain;
                participant.domain_participant_index = static_cast<int>(participant_index);
                participant.address = address;
                participant.device = address.toLocalDeviceId();
                participant.world_rank = participantWorldRank(domain, participant_index);
                participant.world_rank_known = participantWorldRankKnown(domain, participant_index);

                participants.push_back(std::move(participant));
            }
            return participants;
        }

        std::vector<int> orderedExpertIdsForTier(
            const RoutedExpertLayerPlacement &placement,
            int tier_idx,
            RoutedExpertOwnerOrder owner_order)
        {
            std::vector<int> experts;
            for (size_t expert_id = 0; expert_id < placement.routed_expert_tier.size(); ++expert_id)
            {
                if (placement.routed_expert_tier[expert_id] == tier_idx)
                    experts.push_back(static_cast<int>(expert_id));
            }
            std::sort(experts.begin(), experts.end());
            routed_expert_ownership::applyOwnerOrder(
                experts,
                owner_order,
                placement.layer,
                tier_idx);
            return experts;
        }

        bool sameParticipantIdentity(
            const MoEExpertOwnerParticipant &candidate,
            const MoEExpertOwnerParticipant &previous)
        {
            return candidate.participant_id == previous.participant_id &&
                   candidate.tier_idx == previous.tier_idx &&
                   candidate.tier_name == previous.tier_name &&
                   candidate.domain_name == previous.domain_name &&
                   candidate.domain_participant_index ==
                       previous.domain_participant_index &&
                   candidate.address == previous.address &&
                   candidate.device == previous.device &&
                   candidate.world_rank == previous.world_rank &&
                   candidate.world_rank_known ==
                       previous.world_rank_known;
        }

        void requireCompatiblePreviousMap(
            const MoERoutedExpertPlacementPlan &plan,
            const MoEExpertOwnerMap &candidate,
            const MoEExpertOwnerMap &previous)
        {
            if (candidate.participants().size() !=
                previous.participants().size())
            {
                throw std::invalid_argument(
                    "MoE owner-map transition changes participant topology");
            }
            for (size_t index = 0; index < candidate.participants().size();
                 ++index)
            {
                if (!sameParticipantIdentity(
                        candidate.participants()[index],
                        previous.participants()[index]))
                {
                    std::ostringstream message;
                    message << "MoE owner-map transition changes participant "
                            << index << " identity";
                    throw std::invalid_argument(message.str());
                }
            }

            size_t expected_owner_count = 0;
            for (const auto &placement : plan.placements)
            {
                expected_owner_count +=
                    placement.routed_expert_tier.size();
                for (size_t expert_id = 0;
                     expert_id < placement.routed_expert_tier.size();
                     ++expert_id)
                {
                    if (!previous.ownerFor(
                            placement.layer,
                            static_cast<int>(expert_id)))
                    {
                        std::ostringstream message;
                        message
                            << "MoE owner-map transition previous epoch lacks layer "
                            << placement.layer << " expert " << expert_id;
                        throw std::invalid_argument(message.str());
                    }
                }
            }
            if (previous.owners().size() != expected_owner_count)
            {
                throw std::invalid_argument(
                    "MoE owner-map transition changes model ownership geometry");
            }
        }

        MoEExpertOwner makeOwner(
            const RoutedExpertLayerPlacement &placement,
            int expert_id,
            int tier_idx,
            int owner_participant,
            const MoEExpertOwnerParticipant &participant)
        {
            MoEExpertOwner owner;
            owner.layer_idx = placement.layer;
            owner.expert_id = expert_id;
            owner.tier_idx = tier_idx;
            owner.owner_participant = owner_participant;
            owner.device = participant.device;
            owner.resident = true;
            owner.tier_name = participant.tier_name;
            owner.domain_name = participant.domain_name;
            owner.domain_participant_index =
                participant.domain_participant_index;
            owner.owner_world_rank = participant.world_rank;
            owner.owner_world_rank_known = participant.world_rank_known;
            owner.address = participant.address;
            return owner;
        }

        std::vector<MoEExpertOwner> buildLayerTierOwners(
            const MoEExpertOwnerMap &owner_map,
            const RoutedExpertLayerPlacement &placement,
            int tier_idx,
            RoutedExpertOwnerOrder owner_order,
            const MoEExpertOwnerMap *previous)
        {
            const auto expert_ids = orderedExpertIdsForTier(
                placement,
                tier_idx,
                owner_order);
            if (expert_ids.empty())
                return {};

            const auto participant_ids = owner_map.participantIdsForTier(tier_idx);
            if (participant_ids.empty())
            {
                std::ostringstream message;
                message << "MoE expert owner map tier " << tier_idx
                        << " has assigned experts but no domain participants";
                throw std::invalid_argument(message.str());
            }

            const size_t participant_count = participant_ids.size();
            const size_t base_count = expert_ids.size() / participant_count;
            const size_t remainder = expert_ids.size() % participant_count;
            std::vector<size_t> target_counts(participant_count, base_count);
            for (size_t offset = 0; offset < remainder; ++offset)
                ++target_counts[offset];

            std::vector<std::vector<int>> assigned_experts(participant_count);
            std::vector<bool> assigned(expert_ids.size(), false);

            if (previous)
            {
                /*
                 * Retention is the primary transition invariant. Iterating the
                 * canonical expert order makes the rare over-capacity case
                 * deterministic when a tier's total population changes.
                 */
                for (size_t expert_offset = 0;
                     expert_offset < expert_ids.size(); ++expert_offset)
                {
                    const int expert_id = expert_ids[expert_offset];
                    const auto *old_owner = previous->ownerFor(
                        placement.layer,
                        expert_id);
                    if (!old_owner || old_owner->tier_idx != tier_idx)
                        continue;

                    const auto participant_it = std::find(
                        participant_ids.begin(),
                        participant_ids.end(),
                        old_owner->owner_participant);
                    if (participant_it == participant_ids.end())
                    {
                        throw std::logic_error(
                            "MoE retained expert references a participant outside its tier");
                    }
                    const size_t participant_offset =
                        static_cast<size_t>(std::distance(
                            participant_ids.begin(), participant_it));
                    if (assigned_experts[participant_offset].size() >=
                        target_counts[participant_offset])
                    {
                        continue;
                    }
                    assigned_experts[participant_offset].push_back(expert_id);
                    assigned[expert_offset] = true;
                }
            }

            /*
             * New arrivals and unavoidable balance corrections consume exact
             * participant deficits. No already-retained expert is displaced.
             */
            size_t participant_cursor = 0;
            for (size_t expert_offset = 0;
                 expert_offset < expert_ids.size(); ++expert_offset)
            {
                if (assigned[expert_offset])
                    continue;
                while (participant_cursor < participant_count &&
                       assigned_experts[participant_cursor].size() >=
                           target_counts[participant_cursor])
                {
                    ++participant_cursor;
                }
                if (participant_cursor >= participant_count)
                {
                    throw std::logic_error(
                        "MoE stable owner assignment exhausted participant capacity");
                }
                assigned_experts[participant_cursor].push_back(
                    expert_ids[expert_offset]);
            }

            std::vector<MoEExpertOwner> owners;
            owners.reserve(expert_ids.size());

            for (size_t participant_offset = 0; participant_offset < participant_count; ++participant_offset)
            {
                const int owner_participant = participant_ids[participant_offset];
                const auto *participant = owner_map.participantForId(owner_participant);
                if (!participant)
                    throw std::logic_error("MoE expert owner map has a missing participant descriptor");

                if (assigned_experts[participant_offset].size() !=
                    target_counts[participant_offset])
                {
                    throw std::logic_error(
                        "MoE stable owner assignment did not fill exact participant capacity");
                }
                for (const int expert_id :
                     assigned_experts[participant_offset])
                {
                    owners.push_back(makeOwner(
                        placement,
                        expert_id,
                        tier_idx,
                        owner_participant,
                        *participant));
                }
            }
            return owners;
        }

    } // namespace

    MoEExpertOwnerMap MoEExpertOwnerMap::build(
        const MoERoutedExpertPlacementPlan &plan,
        const MoEExpertOwnerMapBuildOptions &options)
    {
        return buildWithPreferredOwners(plan, options, nullptr);
    }

    MoEExpertOwnerMap MoEExpertOwnerMap::buildTransition(
        const MoERoutedExpertPlacementPlan &plan,
        const MoEExpertOwnerMap &previous,
        const MoEExpertOwnerMapBuildOptions &options)
    {
        return buildWithPreferredOwners(plan, options, &previous);
    }

    MoEExpertOwnerMap MoEExpertOwnerMap::buildExplicit(
        const MoERoutedExpertPlacementPlan &plan,
        const MoELayeredExpertOwnership &ownership,
        const MoEExpertOwnerMapBuildOptions &options)
    {
        /*
         * Build the canonical participant vocabulary and validate the logical
         * plan through the ordinary path. We then replace only the derived
         * cold-start owner rows with the planner's explicit physical choices.
         */
        MoEExpertOwnerMap owner_map =
            buildWithPreferredOwners(plan, options, nullptr);
        if (ownership.participantCount() !=
            static_cast<int>(owner_map.participants_.size()))
        {
            throw std::invalid_argument(
                "Explicit MoE ownership participant geometry does not match the tier topology");
        }
        if (ownership.expertCount() <= 0)
        {
            throw std::invalid_argument(
                "Explicit MoE ownership has no routed experts");
        }

        std::vector<bool> represented_layers(
            static_cast<std::size_t>(ownership.layerCount()), false);
        std::vector<MoEExpertOwner> explicit_owners;
        for (const auto &placement : plan.placements)
        {
            if (placement.layer < 0 ||
                placement.layer >= ownership.layerCount())
            {
                throw std::invalid_argument(
                    "Explicit MoE ownership does not contain a planned layer");
            }
            if (represented_layers[static_cast<std::size_t>(placement.layer)])
            {
                throw std::invalid_argument(
                    "Explicit MoE ownership plan repeats a routed layer");
            }
            represented_layers[static_cast<std::size_t>(placement.layer)] = true;
            if (placement.routed_expert_tier.size() !=
                static_cast<std::size_t>(ownership.expertCount()))
            {
                throw std::invalid_argument(
                    "Explicit MoE ownership expert geometry does not match tier placement");
            }

            for (int expert_id = 0;
                 expert_id < ownership.expertCount();
                 ++expert_id)
            {
                const int participant_id = ownership.owner(
                    placement.layer, expert_id);
                const auto *participant =
                    owner_map.participantForId(participant_id);
                if (!participant)
                {
                    throw std::invalid_argument(
                        "Explicit MoE ownership references an unknown participant");
                }
                const int tier_idx = placement.routed_expert_tier[
                    static_cast<std::size_t>(expert_id)];
                if (participant->tier_idx != tier_idx)
                {
                    std::ostringstream message;
                    message
                        << "Explicit MoE owner participant "
                        << participant_id << " belongs to tier "
                        << participant->tier_idx << " but layer "
                        << placement.layer << " expert " << expert_id
                        << " is placed in tier " << tier_idx;
                    throw std::invalid_argument(message.str());
                }
                explicit_owners.push_back(makeOwner(
                    placement,
                    expert_id,
                    tier_idx,
                    participant_id,
                    *participant));
            }
        }

        if (std::find(
                represented_layers.begin(),
                represented_layers.end(),
                false) != represented_layers.end())
        {
            throw std::invalid_argument(
                "Explicit MoE ownership contains a layer absent from tier placement");
        }

        owner_map.owners_ = std::move(explicit_owners);
        return owner_map;
    }

    MoEExpertOwnerMap MoEExpertOwnerMap::buildWithPreferredOwners(
        const MoERoutedExpertPlacementPlan &plan,
        const MoEExpertOwnerMapBuildOptions &options,
        const MoEExpertOwnerMap *previous)
    {
        if (!plan.usesExpertOverlayAuthority())
            throw std::invalid_argument(
                "MoEExpertOwnerMap requires an enabled tiered routed-expert placement plan");
        if (plan.placements.empty())
            throw std::invalid_argument("MoEExpertOwnerMap requires explicit layer expert placements");

        const auto validation = validateMoERoutedExpertPlacementPlan(plan);
        if (!validation.ok())
            throw std::invalid_argument(formatValidationErrors(validation));

        MoEExpertOwnerMap owner_map;
        for (size_t tier_index = 0; tier_index < plan.routed_tiers.size(); ++tier_index)
        {
            auto participants = buildTierParticipants(
                plan,
                tier_index,
                options,
                static_cast<int>(owner_map.participants_.size()));
            owner_map.participants_.insert(
                owner_map.participants_.end(),
                std::make_move_iterator(participants.begin()),
                std::make_move_iterator(participants.end()));
        }

        if (previous)
            requireCompatiblePreviousMap(plan, owner_map, *previous);

        for (const auto &placement : plan.placements)
        {
            for (size_t tier_index = 0; tier_index < plan.routed_tiers.size(); ++tier_index)
            {
                auto owners = buildLayerTierOwners(
                    owner_map,
                    placement,
                    static_cast<int>(tier_index),
                    plan.owner_order,
                    previous);
                owner_map.owners_.insert(
                    owner_map.owners_.end(),
                    std::make_move_iterator(owners.begin()),
                    std::make_move_iterator(owners.end()));
            }
        }

        std::map<std::pair<int, int>, int> owner_counts;
        for (const auto &owner : owner_map.owners_)
            ++owner_counts[{owner.layer_idx, owner.expert_id}];

        for (const auto &placement : plan.placements)
        {
            for (size_t expert_id = 0; expert_id < placement.routed_expert_tier.size(); ++expert_id)
            {
                const auto key = std::make_pair(placement.layer, static_cast<int>(expert_id));
                const auto found = owner_counts.find(key);
                const int count = found == owner_counts.end() ? 0 : found->second;
                if (count != 1)
                {
                    std::ostringstream message;
                    message << "MoE expert owner map produced " << count
                            << " owners for layer " << placement.layer
                            << " expert " << expert_id;
                    throw std::logic_error(message.str());
                }
            }
        }

        for (const auto &[key, count] : owner_counts)
        {
            if (count != 1)
            {
                std::ostringstream message;
                message << "MoE expert owner map produced " << count
                        << " owners for layer " << key.first
                        << " expert " << key.second;
                throw std::logic_error(message.str());
            }
        }

        return owner_map;
    }

    const MoEExpertOwner *MoEExpertOwnerMap::ownerFor(int layer_idx, int expert_id) const
    {
        auto it = std::find_if(owners_.begin(), owners_.end(), [&](const auto &owner)
                               { return owner.layer_idx == layer_idx && owner.expert_id == expert_id; });
        return it == owners_.end() ? nullptr : &(*it);
    }

    const MoEExpertOwnerParticipant *MoEExpertOwnerMap::participantForId(int participant_id) const
    {
        auto it = std::find_if(participants_.begin(), participants_.end(), [&](const auto &participant)
                               { return participant.participant_id == participant_id; });
        return it == participants_.end() ? nullptr : &(*it);
    }

    std::vector<int> MoEExpertOwnerMap::participantIdsForTier(int tier_idx) const
    {
        std::vector<int> ids;
        for (const auto &participant : participants_)
        {
            if (participant.tier_idx == tier_idx)
                ids.push_back(participant.participant_id);
        }
        return ids;
    }

    std::vector<int> MoEExpertOwnerMap::expertsForParticipant(
        int layer_idx,
        int owner_participant) const
    {
        std::vector<int> experts;
        for (const auto &owner : owners_)
        {
            if (owner.layer_idx == layer_idx && owner.owner_participant == owner_participant)
                experts.push_back(owner.expert_id);
        }
        std::sort(experts.begin(), experts.end());
        return experts;
    }

    std::vector<bool> MoEExpertOwnerMap::expertMaskForParticipant(
        int layer_idx,
        int owner_participant,
        int num_experts) const
    {
        std::vector<bool> mask(static_cast<size_t>(std::max(0, num_experts)), false);
        for (const int expert_id : expertsForParticipant(layer_idx, owner_participant))
        {
            if (expert_id >= 0 && expert_id < num_experts)
                mask[static_cast<size_t>(expert_id)] = true;
        }
        return mask;
    }

    size_t MoEExpertOwnerMap::ownerCountForExpert(int layer_idx, int expert_id) const
    {
        return static_cast<size_t>(std::count_if(owners_.begin(), owners_.end(), [&](const auto &owner)
                                                 { return owner.layer_idx == layer_idx && owner.expert_id == expert_id; }));
    }

    MoELayeredExpertOwnership MoEExpertOwnerMap::layeredOwnership(
        int num_layers,
        int num_experts) const
    {
        const int participant_count = static_cast<int>(participants_.size());
        if (num_layers <= 0 || num_experts <= 0 || participant_count <= 0)
        {
            throw std::invalid_argument(
                "MoE expert owner map requires positive model geometry and at least one participant");
        }

        std::vector<std::vector<int>> owners(
            static_cast<size_t>(num_layers),
            std::vector<int>(static_cast<size_t>(num_experts), -1));
        for (const auto &owner : owners_)
        {
            if (owner.layer_idx < 0 || owner.layer_idx >= num_layers ||
                owner.expert_id < 0 || owner.expert_id >= num_experts ||
                owner.owner_participant < 0 ||
                owner.owner_participant >= participant_count)
            {
                throw std::logic_error(
                    "MoE expert owner map exceeds the requested model or participant geometry");
            }

            int &slot = owners[static_cast<size_t>(owner.layer_idx)]
                              [static_cast<size_t>(owner.expert_id)];
            if (slot >= 0)
            {
                throw std::logic_error(
                    "MoE expert owner map contains duplicate layered ownership");
            }
            slot = owner.owner_participant;
        }

        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx)
        {
            for (int expert_id = 0; expert_id < num_experts; ++expert_id)
            {
                if (owners[static_cast<size_t>(layer_idx)]
                          [static_cast<size_t>(expert_id)] < 0)
                {
                    throw std::logic_error(
                        "MoE expert owner map is incomplete for the requested model geometry");
                }
            }
        }

        return MoELayeredExpertOwnership(
            participant_count,
            std::move(owners));
    }

} // namespace llaminar2
