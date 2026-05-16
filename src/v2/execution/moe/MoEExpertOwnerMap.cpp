/**
 * @file MoEExpertOwnerMap.cpp
 * @brief Graph-native routed MoE whole-expert owner map implementation.
 */

#include "MoEExpertOwnerMap.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        std::string formatValidationErrors(const MoEExpertParallelValidationResult &validation)
        {
            std::ostringstream message;
            message << "Invalid MoE expert owner map plan:";
            for (const auto &error : validation.errors)
                message << "\n - " << error;
            return message.str();
        }

        const ExpertComputeDomain &requireDomain(
            const MoEExpertParallelPlan &plan,
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

        int participantWorldRank(const ExpertComputeDomain &domain, size_t participant_index)
        {
            if (participant_index < domain.world_ranks.size())
                return domain.world_ranks[participant_index];
            if (domain.kind == ExpertDomainKind::NodeLocalTP)
                return static_cast<int>(participant_index);
            if (domain.owner_rank >= 0)
                return domain.owner_rank;
            return -1;
        }

        bool participantWorldRankKnown(const ExpertComputeDomain &domain, size_t participant_index)
        {
            return participant_index < domain.world_ranks.size() ||
                   domain.kind == ExpertDomainKind::NodeLocalTP ||
                   domain.owner_rank >= 0;
        }

        std::vector<MoEExpertOwnerParticipant> buildTierParticipants(
            const MoEExpertParallelPlan &plan,
            size_t tier_index,
            const MoEExpertOwnerMapBuildOptions &options,
            int first_participant_id)
        {
            const auto &tier = plan.routed_tiers[tier_index];
            const auto &domain = requireDomain(plan, tier.domain, "routed tier");

            if (options.reject_tensor_parallel_experts &&
                domain.compute_kind == ExpertDomainComputeKind::TensorParallelExperts)
            {
                std::ostringstream message;
                message << "Graph-native routed tier '" << tier.name << "' in domain '" << domain.name
                        << "' requests TensorParallelExperts; Phase 4 whole-expert owner maps reject routed expert GEMM sharding";
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

        std::vector<int> sortedExpertIdsForTier(const ExpertLayerPlacement &placement, int tier_idx)
        {
            std::vector<int> experts;
            for (size_t expert_id = 0; expert_id < placement.routed_expert_tier.size(); ++expert_id)
            {
                if (placement.routed_expert_tier[expert_id] == tier_idx)
                    experts.push_back(static_cast<int>(expert_id));
            }
            std::sort(experts.begin(), experts.end());
            return experts;
        }

        std::vector<MoEExpertOwner> buildLayerTierOwners(
            const MoEExpertOwnerMap &owner_map,
            const ExpertLayerPlacement &placement,
            int tier_idx)
        {
            const auto expert_ids = sortedExpertIdsForTier(placement, tier_idx);
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
            size_t expert_cursor = 0;
            std::vector<MoEExpertOwner> owners;
            owners.reserve(expert_ids.size());

            for (size_t participant_offset = 0; participant_offset < participant_count; ++participant_offset)
            {
                const size_t count = base_count + (participant_offset < remainder ? 1u : 0u);
                const int owner_participant = participant_ids[participant_offset];
                const auto *participant = owner_map.participantForId(owner_participant);
                if (!participant)
                    throw std::logic_error("MoE expert owner map has a missing participant descriptor");

                for (size_t assigned = 0; assigned < count && expert_cursor < expert_ids.size(); ++assigned, ++expert_cursor)
                {
                    MoEExpertOwner owner;
                    owner.layer_idx = placement.layer;
                    owner.expert_id = expert_ids[expert_cursor];
                    owner.tier_idx = tier_idx;
                    owner.owner_participant = owner_participant;
                    owner.device = participant->device;
                    owner.resident = true;
                    owner.tier_name = participant->tier_name;
                    owner.domain_name = participant->domain_name;
                    owner.domain_participant_index = participant->domain_participant_index;
                    owner.owner_world_rank = participant->world_rank;
                    owner.owner_world_rank_known = participant->world_rank_known;
                    owner.address = participant->address;

                    owners.push_back(std::move(owner));
                }
            }
            return owners;
        }

    } // namespace

    MoEExpertOwnerMap MoEExpertOwnerMap::build(
        const MoEExpertParallelPlan &plan,
        const MoEExpertOwnerMapBuildOptions &options)
    {
        if (!plan.isTieredOverlay())
            throw std::invalid_argument("MoEExpertOwnerMap requires an enabled TieredExpertOverlay plan");
        if (plan.placements.empty())
            throw std::invalid_argument("MoEExpertOwnerMap requires explicit layer expert placements");

        const auto validation = validateMoEExpertParallelPlan(
            plan,
            MoEExpertParallelValidationOptions{
                .allow_routed_tensor_parallel_experts = !options.reject_tensor_parallel_experts,
            });
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

        for (const auto &placement : plan.placements)
        {
            for (size_t tier_index = 0; tier_index < plan.routed_tiers.size(); ++tier_index)
            {
                auto owners = buildLayerTierOwners(owner_map, placement, static_cast<int>(tier_index));
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

} // namespace llaminar2