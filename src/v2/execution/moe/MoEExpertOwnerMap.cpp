/**
 * @file MoEExpertOwnerMap.cpp
 * @brief Deterministic initial and epoch-transition routed-expert assignment.
 *
 * Logical tier placement is converted to exact physical ownership here. The
 * transition path first retains compatible prior owners and then fills only
 * participant deficits in canonical order. That rule avoids gratuitous
 * same-tier copies while keeping independently planning MPI ranks bit-for-bit
 * deterministic.
 * Migration vertices preserve complete shard cohorts, rather than aliasing a
 * jointly owned expert to one member or its absent scalar-owner sentinel.
 */

#include <atomic>
#include "utils/PerfStatsCollector.h"
#include "MoEExpertOwnerMap.h"
#include "MoEExpertShardCohort.h"
#include "RoutedExpertOwnerAssignment.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <set>
#include <tuple>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        std::atomic<std::uint64_t> initial_build_attempts{0};

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
                domain.routed_compute_policy == RoutedExpertComputePolicy::TensorSharded &&
                !(domain.participants.size()==1 && domain.participants.front().isCPU()))
            {
                std::ostringstream message;
                message << "Graph-native routed tier '" << tier.name << "' in domain '" << domain.name
                        << "' uses expert_compute=tensor-sharded; a whole-expert owner map cannot represent within-expert tensor shards";
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

        /**
         * @brief Put a complete owner relation in construction-independent order.
         *
         * Transition assignment is naturally produced tier-by-tier while an
         * explicit assignment is naturally produced expert-by-expert.  Those
         * are implementation details, not residency state.  Sorting once at
         * setup/maintenance time gives every consumer one canonical value and
         * prevents distributed fingerprints from depending on the builder.
         */
        void canonicalizeOwnerOrder(std::vector<MoEExpertOwner> &owners)
        {
            std::sort(
                owners.begin(),
                owners.end(),
                [](const MoEExpertOwner &lhs, const MoEExpertOwner &rhs)
                {
                    if (lhs.layer_idx != rhs.layer_idx)
                        return lhs.layer_idx < rhs.layer_idx;
                    return lhs.expert_id < rhs.expert_id;
                });
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
                    const auto *old_owner = previous->placementFor(placement.layer, expert_id);
                    if (!old_owner || old_owner->tier_idx != tier_idx)
                        continue;
                    if (old_owner->shard_cohort)
                        throw std::logic_error(
                            "A retained CPU cohort cannot become a scalar whole owner");

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

    bool MoEExpertOwner::operator==(const MoEExpertOwner& other) const
    {
        if(std::tie(layer_idx,expert_id,tier_idx,owner_participant,device,resident,
               tier_name,domain_name,domain_participant_index,owner_world_rank,owner_world_rank_known,address)!=
           std::tie(other.layer_idx,other.expert_id,other.tier_idx,other.owner_participant,other.device,other.resident,
               other.tier_name,other.domain_name,other.domain_participant_index,other.owner_world_rank,other.owner_world_rank_known,other.address))
            return false;
        if(shard_cohort==other.shard_cohort)return true;
        return shard_cohort && other.shard_cohort && *shard_cohort==*other.shard_cohort;
    }

    std::optional<MoEExpertPlacementTarget> MoEExpertPlacementTarget::fromPlacement(
        const MoEExpertOwner &placement) noexcept
    {
        if (placement.layer_idx < 0 || placement.expert_id < 0 || placement.tier_idx < 0)
            return std::nullopt;
        if (!placement.shard_cohort)
        {
            if (placement.owner_participant < 0)
                return std::nullopt;
            return MoEExpertPlacementTarget(placement.tier_idx, placement.owner_participant);
        }
        const auto &cohort = *placement.shard_cohort;
        if (cohort.participants().size() < 2 || cohort.participants().size() > 8 ||
            placement.owner_participant != -1 || placement.device.is_valid() ||
            placement.owner_world_rank_known || placement.owner_world_rank != -1 ||
            placement.domain_participant_index != -1 ||
            cohort.source().layer_idx != placement.layer_idx ||
            placement.expert_id >= cohort.ownership(0).geometry().experts)
            return std::nullopt;
        return MoEExpertPlacementTarget(placement.tier_idx, placement.shard_cohort);
    }

    bool MoEExpertPlacementTarget::isCohort() const noexcept
    {
        return std::holds_alternative<std::shared_ptr<const MoEExpertShardCohort>>(owners_);
    }

    std::size_t MoEExpertPlacementTarget::participantCount() const noexcept
    {
        return isCohort()
            ? std::get<std::shared_ptr<const MoEExpertShardCohort>>(owners_)->participants().size()
            : 1u;
    }

    int MoEExpertPlacementTarget::participantId(std::size_t index) const
    {
        if (index >= participantCount())
            throw std::out_of_range("Expert placement target has no such original physical member");
        return isCohort()
            ? std::get<std::shared_ptr<const MoEExpertShardCohort>>(owners_)->participants()[index].participant_id
            : std::get<int>(owners_);
    }

    bool MoEExpertPlacementTarget::operator==(const MoEExpertPlacementTarget &other) const noexcept
    {
        if (tier_idx_ != other.tier_idx_ || owners_.index() != other.owners_.index())
            return false;
        if (!isCohort())
            return std::get<int>(owners_) == std::get<int>(other.owners_);
        const auto &ours = std::get<std::shared_ptr<const MoEExpertShardCohort>>(owners_);
        const auto &theirs = std::get<std::shared_ptr<const MoEExpertShardCohort>>(other.owners_);
        return ours == theirs || *ours == *theirs;
    }

    std::uint64_t MoEExpertOwnerMap::initialBuildAttempts() noexcept
    {
        return initial_build_attempts.load(std::memory_order_relaxed);
    }

    bool MoEExpertOwnerMap::usesCPUJointFragments(const MoERoutedExpertPlacementPlan& plan) noexcept {
        for(const auto &tier:plan.routed_tiers) {
            const auto domain=std::find_if(plan.domains.begin(),plan.domains.end(),
                [&](const auto& candidate){return candidate.name==tier.domain;});
            if(domain!=plan.domains.end() && domain->routed_compute_policy==RoutedExpertComputePolicy::TensorSharded &&
               domain->participants.size()>1 && std::all_of(domain->participants.begin(),domain->participants.end(),
                   [](const auto& address){return address.isCPU();}))return true;
        }
        return false;
    }

    std::vector<MoEExpertOwnerParticipant> MoEExpertOwnerMap::declaredParticipants(
        const MoERoutedExpertPlacementPlan& plan,const MoEExpertOwnerMapBuildOptions& options)
    {
        if(!plan.usesExpertOverlayAuthority())throw std::invalid_argument("Routed participants require original enabled tiered topology");
        std::vector<MoEExpertOwnerParticipant> result;
        for(std::size_t tier=0;tier<plan.routed_tiers.size();++tier){
            const auto& domain=requireDomain(plan,plan.routed_tiers[tier].domain,"routed tier");
            if(domain.participants.size()>static_cast<std::size_t>(std::numeric_limits<int>::max())-result.size())
                throw std::overflow_error("Routed participant identity extent overflows");
            auto next=buildTierParticipants(plan,tier,options,static_cast<int>(result.size()));
            if(result.size()+next.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::overflow_error("Routed participant identity extent overflows");
            result.insert(result.end(),std::make_move_iterator(next.begin()),std::make_move_iterator(next.end()));
        }
        return result;
    }

    MoEExpertOwnerMap MoEExpertOwnerMap::build(
        const MoERoutedExpertPlacementPlan &plan,
        const MoEExpertOwnerMapBuildOptions &options)
    {
        initial_build_attempts.fetch_add(1, std::memory_order_relaxed);
        PerfStatsCollector::addCounter("forward_graph", "canonical_owner_map_build_attempts", 1.0,
            "setup", {}, {{"reject_tensor_sharded_domains", options.reject_tensor_sharded_domains ? "true" : "false"}});
        return buildWithPreferredOwners(plan, options, nullptr);
    }

    MoEExpertOwnerMap MoEExpertOwnerMap::buildShardPlacement(
        const MoERoutedExpertPlacementPlan& plan,
        const std::vector<MoEOverlayLayerWeightManifest>& manifest)
    {
        initial_build_attempts.fetch_add(1,std::memory_order_relaxed);
        PerfStatsCollector::addCounter("forward_graph","canonical_owner_map_build_attempts",1.0,
            "setup",{},{{"ownership_kind","joint_tensor_shards"}});
        const auto validation=validateMoERoutedExpertPlacementPlan(plan);
        if(!plan.usesExpertOverlayAuthority() || plan.placements.empty() || !validation.ok())
            throw std::invalid_argument("CPU shard placement requires complete original tier placements");
        MoEExpertOwnerMap result;
        result.participants_=declaredParticipants(plan,{.reject_tensor_sharded_domains=false});
        const auto experts=plan.placements.front().routed_expert_tier.size();
        if(experts==0 || experts>static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("CPU shard placement requires bounded original expert IDs");
        std::set<int> source_layers;
        for(const auto& source:manifest)if(!source.valid() || !source_layers.insert(source.layer_idx).second)
            throw std::invalid_argument("CPU shard placement source is invalid or duplicated");
        if(source_layers.size()!=plan.placements.size())
            throw std::invalid_argument("CPU shard placement contains foreign or omitted source layers");
        for(const auto& placement:plan.placements){
            if(placement.routed_expert_tier.size()!=experts)
                throw std::invalid_argument("CPU shard placement changed the original expert axis");
            const auto source=std::find_if(manifest.begin(),manifest.end(),[&](const auto& layer){return layer.layer_idx==placement.layer;});
            if(source==manifest.end())throw std::invalid_argument("CPU shard placement omitted an original source layer");
            for(std::size_t tier=0;tier<plan.routed_tiers.size();++tier){
                const auto& domain=requireDomain(plan,plan.routed_tiers[tier].domain,"shard placement");
                if(domain.routed_compute_policy!=RoutedExpertComputePolicy::TensorSharded ||
                   (domain.participants.size()==1 && domain.participants.front().isCPU())){
                    auto owners=buildLayerTierOwners(result,placement,static_cast<int>(tier),plan.owner_order,nullptr);
                    result.owners_.insert(result.owners_.end(),std::make_move_iterator(owners.begin()),std::make_move_iterator(owners.end()));
                    continue;
                }
                if(domain.scope!=ExecutionDomainScope::NODE_LOCAL ||
                   domain.world_ranks.size()!=domain.participants.size())
                    throw std::invalid_argument("CPU shard placement requires original inventory-bound node-local CPU ranks");
                std::vector<MoEExpertShardCohort::Participant> members;
                for(const auto id:result.participantIdsForTier(static_cast<int>(tier))){
                    const auto& p=*result.participantForId(id);
                    if(!p.world_rank_known)throw std::invalid_argument("CPU shard placement requires bound original ranks");
                    members.push_back({id,p.world_rank,p.address});
                }
                auto cohort=std::make_shared<const MoEExpertShardCohort>(MoEExpertShardCohort::checked(
                    *source,static_cast<int>(experts),std::move(members),
                    MoEExpertShardCohort::balancedIntervals(*source,static_cast<int>(domain.participants.size()))));
                for(std::size_t expert=0;expert<experts;++expert)if(placement.routed_expert_tier[expert]==static_cast<int>(tier)){
                    MoEExpertOwner row;row.layer_idx=placement.layer;row.expert_id=static_cast<int>(expert);row.tier_idx=static_cast<int>(tier);
                    row.owner_participant=-1;row.device=DeviceId::invalid();row.resident=true;
                    row.tier_name=plan.routed_tiers[tier].name;row.domain_name=domain.name;row.shard_cohort=cohort;
                    result.owners_.push_back(std::move(row));
                }
            }
        }
        canonicalizeOwnerOrder(result.owners_);
        if(result.owners_.size()!=plan.placements.size()*experts)
            throw std::logic_error("CPU shard placement lost original logical expert coverage");
        return result;
    }

    MoEExpertOwnerMap
    MoEExpertOwnerMap::buildShardTransition(const MoERoutedExpertPlacementPlan &plan,
                                            const MoEExpertOwnerMap &previous,
                                            const MoEExpertOwnerMap &original,
                                            std::span<const MoELayeredExpertOwnershipChange> whole_owner_changes)
    {
        const auto validation = validateMoERoutedExpertPlacementPlan(plan);
        if (!plan.usesExpertOverlayAuthority() || !usesCPUJointFragments(plan) ||
            plan.placements.empty() || !validation.ok())
            throw std::invalid_argument(
                "Joint placement transition requires complete unchanged source topology");
        MoEExpertOwnerMap result;
        result.participants_ = declaredParticipants(plan, {.reject_tensor_sharded_domains = false});
        if (result.participants_ != original.participants_ ||
            result.participants_ != previous.participants_)
            throw std::invalid_argument(
                "Joint placement transition changed original participant topology");
        std::size_t total = 0;
        for (const auto &placement : plan.placements)
        {
            const auto experts = placement.routed_expert_tier.size();
            total += experts;
            // These short-lived aliases accelerate this layer's construction;
            // the original map remains the sole source of cohort identity.
            std::vector<std::shared_ptr<const MoEExpertShardCohort>> cohorts(
                plan.routed_tiers.size());
            std::vector<std::size_t> source_counts(plan.routed_tiers.size(), 0);
            std::vector<std::size_t> previous_counts(plan.routed_tiers.size(), 0);
            std::vector<std::size_t> candidate_counts(plan.routed_tiers.size(), 0);
            const auto checked_tier = [&](const MoEExpertOwner *row, int expert) -> std::size_t
            {
                if (!row || row->layer_idx != placement.layer || row->expert_id != expert ||
                    !row->resident || row->tier_idx < 0 ||
                    std::size_t(row->tier_idx) >= plan.routed_tiers.size() ||
                    !MoEExpertPlacementTarget::fromPlacement(*row))
                    throw std::invalid_argument(
                        "Joint placement transition has foreign original layer/expert coordinates");
                return std::size_t(row->tier_idx);
            };
            for (std::size_t expert = 0; expert < experts; ++expert)
            {
                const auto *source = original.placementFor(placement.layer, int(expert));
                const auto tier = checked_tier(source, int(expert));
                const auto &domain =
                    requireDomain(plan, plan.routed_tiers[tier].domain, "joint transition");
                const bool joint =
                    domain.routed_compute_policy == RoutedExpertComputePolicy::TensorSharded &&
                    domain.participants.size() > 1 && domain.participants.front().isCPU();
                if (joint != bool(source->shard_cohort))
                    throw std::invalid_argument(
                        "Joint placement transition changed original projection ownership");
                if (source->shard_cohort)
                {
                    auto &retained = cohorts[tier];
                    if (retained && (retained.get() != source->shard_cohort.get() ||
                                     retained.owner_before(source->shard_cohort) ||
                                     source->shard_cohort.owner_before(retained)))
                        throw std::invalid_argument(
                            "Joint source tier contains different original cohort owners");
                    retained = source->shard_cohort;
                }
                ++source_counts[tier];
            }
            for (std::size_t expert = 0; expert < experts; ++expert)
            {
                const auto *old = previous.placementFor(placement.layer, int(expert));
                const auto tier = checked_tier(old, int(expert));
                if (cohorts[tier])
                {
                    if (old->shard_cohort.get() != cohorts[tier].get() ||
                        old->shard_cohort.owner_before(cohorts[tier]) ||
                        cohorts[tier].owner_before(old->shard_cohort))
                        throw std::invalid_argument(
                            "Joint live placement replaced original source cohort/cuts");
                }
                else if (old->shard_cohort)
                    throw std::invalid_argument(
                        "Joint live placement introduced a foreign source cohort");
                ++previous_counts[tier];
                const int destination = placement.routed_expert_tier[expert];
                if (destination < 0 || std::size_t(destination) >= candidate_counts.size())
                    throw std::invalid_argument(
                        "Joint placement transition has a foreign destination tier");
                ++candidate_counts[std::size_t(destination)];
            }
            if (source_counts != previous_counts || source_counts != candidate_counts)
                throw std::invalid_argument(
                    "Joint placement transition changes original per-layer tier capacity");
            for (std::size_t tier = 0; tier < plan.routed_tiers.size(); ++tier)
            {
                if (!cohorts[tier])
                {
                    auto owners = buildLayerTierOwners(result, placement, int(tier),
                                                       plan.owner_order, &previous);
                    result.owners_.insert(result.owners_.end(),
                                          std::make_move_iterator(owners.begin()),
                                          std::make_move_iterator(owners.end()));
                    continue;
                }
                for (std::size_t expert = 0; expert < experts; ++expert)
                {
                    if (placement.routed_expert_tier[expert] != int(tier))
                        continue;
                    MoEExpertOwner row;
                    row.layer_idx = placement.layer;
                    row.expert_id = int(expert);
                    row.tier_idx = int(tier);
                    row.owner_participant = -1;
                    row.device = DeviceId::invalid();
                    row.resident = true;
                    row.tier_name = plan.routed_tiers[tier].name;
                    row.domain_name = plan.routed_tiers[tier].domain;
                    row.shard_cohort = cohorts[tier];
                    result.owners_.push_back(std::move(row));
                }
            }
        }
        if (original.owners_.size() != total || previous.owners_.size() != total ||
            result.owners_.size() != total)
            throw std::invalid_argument(
                "Joint placement transition changes complete original source geometry");
        // This is an unpublished candidate, not a second live ownership table.
        // Authenticate every explicit whole-owner edge against the canonical tier
        // projection before changing a row. Fixed CPU fragment targets remain
        // source-owned and cannot enter this scalar participant interface.
        std::set<std::pair<int, int>> changed_experts;
        std::map<std::tuple<int, int, int>, int> participant_deltas;
        std::vector<std::pair<std::size_t, const MoEExpertOwnerParticipant *>> replacements;
        replacements.reserve(whole_owner_changes.size());
        for (const auto &change : whole_owner_changes)
        {
            if (change.layer_idx < 0 || change.expert_id < 0 ||
                change.previous_participant < 0 || change.current_participant < 0 ||
                change.previous_participant == change.current_participant ||
                !changed_experts.emplace(change.layer_idx, change.expert_id).second)
                throw std::invalid_argument("Joint whole-owner participant transition has duplicate or invalid change coordinates");
            const auto row = std::find_if(result.owners_.begin(), result.owners_.end(),
                [&](const auto &owner) { return owner.layer_idx == change.layer_idx && owner.expert_id == change.expert_id; });
            if (row == result.owners_.end() || !row->resident || row->shard_cohort ||
                row->owner_participant != change.previous_participant)
                throw std::invalid_argument("Joint whole-owner participant transition requires its exact canonical whole source owner");
            const auto *destination = result.participantForId(change.current_participant);
            const auto &domain = requireDomain(plan, row->domain_name, "joint whole-owner participant transition");
            if (!destination || destination->tier_idx != row->tier_idx ||
                destination->domain_name != row->domain_name ||
                !destination->world_rank_known ||
                domain.routed_compute_policy != RoutedExpertComputePolicy::Apportioned)
                throw std::invalid_argument("Joint whole-owner participant transition requires an original apportioned whole target in the same tier");
            --participant_deltas[{row->layer_idx, row->tier_idx, row->owner_participant}];
            ++participant_deltas[{row->layer_idx, row->tier_idx, destination->participant_id}];
            replacements.emplace_back(static_cast<std::size_t>(row - result.owners_.begin()), destination);
        }
        for (const auto &[coordinate, delta] : participant_deltas)
            if (delta != 0)
                throw std::invalid_argument("Joint whole-owner participant transition changes per-layer participant slot capacity");
        for (const auto &[index, destination] : replacements)
        {
            auto &row = result.owners_[index];
            const auto placement = std::find_if(plan.placements.begin(), plan.placements.end(),
                [&](const auto &value) { return value.layer == row.layer_idx; });
            if (placement == plan.placements.end())
                throw std::logic_error("Joint whole-owner participant transition lost its authenticated layer");
            row = makeOwner(*placement, row.expert_id, row.tier_idx,
                destination->participant_id, *destination);
        }
        canonicalizeOwnerOrder(result.owners_);
        return result;
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
            if (!ownership.containsModelLayer(placement.layer))
            {
                throw std::invalid_argument(
                    "Explicit MoE ownership does not contain a planned layer");
            }
            const auto storage_index =
                ownership.storageIndexForModelLayer(placement.layer);
            if (represented_layers[storage_index])
            {
                throw std::invalid_argument(
                    "Explicit MoE ownership plan repeats a routed layer");
            }
            represented_layers[storage_index] = true;
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

        canonicalizeOwnerOrder(explicit_owners);
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
        owner_map.participants_=declaredParticipants(plan,options);

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

        canonicalizeOwnerOrder(owner_map.owners_);

        return owner_map;
    }

    const MoEExpertOwner *MoEExpertOwnerMap::placementFor(int layer_idx, int expert_id) const
    {
        /*
         * Construction canonicalizes this immutable relation by
         * `(layer, expert)`. Dispatch consults it once per routed activation,
         * so a linear scan turns ordinary prefill into O(routes * model
         * experts) host work. Preserve the general sparse-coordinate contract
         * while using that canonical order as the lookup authority.
         */
        const std::pair<int, int> coordinate{layer_idx, expert_id};
        const auto owner = std::lower_bound(
            owners_.begin(),
            owners_.end(),
            coordinate,
            [](const MoEExpertOwner &candidate,
               const std::pair<int, int> &requested)
            {
                return std::pair{
                           candidate.layer_idx,
                           candidate.expert_id} < requested;
            });
        if (owner == owners_.end() ||
            owner->layer_idx != layer_idx ||
            owner->expert_id != expert_id)
        {
            return nullptr;
        }
        return &*owner;
    }

    const MoEExpertOwner* MoEExpertOwnerMap::ownerFor(int layer_idx,int expert_id) const
    {
        const auto* placement=placementFor(layer_idx,expert_id);
        if(placement && placement->shard_cohort)
            throw std::logic_error("A jointly tensor-sharded expert has no whole-expert owner");
        return placement;
    }

    std::size_t MoEExpertOwnerMap::fragmentCountForExpert(int layer_idx,int expert_id) const
    {
        const auto* placement=placementFor(layer_idx,expert_id);
        return !placement?0:placement->shard_cohort?placement->shard_cohort->participants().size():1;
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
            if (owner.layer_idx == layer_idx &&
                (owner.shard_cohort ? std::any_of(owner.shard_cohort->participants().begin(),
                    owner.shard_cohort->participants().end(),[&](const auto& p){return p.participant_id==owner_participant;})
                 : owner.owner_participant == owner_participant))
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
        /* A successfully built map makes duplicate ownership unrepresentable. */
        return ownerFor(layer_idx, expert_id) ? 1u : 0u;
    }

    void MoEExpertOwnerMap::requireLayerGeometry(
        int num_layers,
        int num_experts,
        int first_model_layer) const
    {
        if (num_layers <= 0 || num_experts <= 0 || participants_.empty() ||
            participants_.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::invalid_argument(
                "MoE expert owner map requires positive model geometry and at least one participant");
        }
        if (first_model_layer < 0 ||
            num_layers > std::numeric_limits<int>::max() - first_model_layer)
        {
            throw std::invalid_argument(
                "MoE expert owner map requires a valid model-layer interval");
        }
        if (static_cast<std::size_t>(num_layers) >
                std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(num_experts))
            throw std::invalid_argument("MoE expert ownership geometry overflows addressable rows");
        if (owners_.size() != static_cast<std::size_t>(num_layers) * static_cast<std::size_t>(num_experts))
            throw std::logic_error("MoE expert owner map is incomplete for the requested model geometry");

        // Construction seals a unique relation in canonical coordinate order.
        // Authenticate that relation directly; validation needs no second live
        // ownership table and cannot accept a same-sized foreign-stage map.
        for (std::size_t index = 0; index < owners_.size(); ++index)
        {
            const auto &owner = owners_[index];
            if (owner.layer_idx != first_model_layer + static_cast<int>(index / num_experts) ||
                owner.expert_id != static_cast<int>(index % num_experts))
            {
                throw std::logic_error(
                    "MoE expert owner map exceeds the requested model or participant geometry");
            }
            const auto target = MoEExpertPlacementTarget::fromPlacement(owner);
            if (!target)
                throw std::logic_error("MoE expert geometry requires a complete physical placement target");
            for (std::size_t member = 0; member < target->participantCount(); ++member)
            {
                const auto *declared = participantForId(target->participantId(member));
                if (!declared || declared->tier_idx != owner.tier_idx)
                    throw std::logic_error("MoE expert geometry names an undeclared physical member");
                if (owner.shard_cohort)
                {
                    const auto &fragment = owner.shard_cohort->participants()[member];
                    if (!declared->device.is_cpu() || !declared->world_rank_known ||
                        declared->world_rank != fragment.world_rank || declared->address != fragment.address)
                        throw std::logic_error("MoE expert geometry changed its original CPU cohort member");
                }
            }
            if (owner.shard_cohort &&
                (owner.shard_cohort->source().layer_idx != owner.layer_idx ||
                 owner.shard_cohort->ownership(0).geometry().experts != num_experts))
                throw std::logic_error("MoE expert cohort has foreign original model geometry");
        }
    }

    MoELayeredExpertOwnership MoEExpertOwnerMap::layeredOwnership(
        int num_layers,
        int num_experts,
        int first_model_layer) const
    {
        requireLayerGeometry(num_layers, num_experts, first_model_layer);
        // Geometry is meaningful for both ownership arms. A scalar execution
        // table is not: never substitute one cohort member (or -1) for the
        // complete target after the common geometry check succeeds.
        if (std::any_of(owners_.begin(), owners_.end(),
                        [](const auto &owner) { return bool(owner.shard_cohort); }))
            throw std::logic_error("Scalar layered ownership cannot represent expert shard cohorts");
        std::vector<std::vector<int>> owners(
            static_cast<std::size_t>(num_layers),
            std::vector<int>(static_cast<std::size_t>(num_experts)));
        for (const auto &owner : owners_)
            owners[static_cast<std::size_t>(owner.layer_idx - first_model_layer)]
                  [static_cast<std::size_t>(owner.expert_id)] = owner.owner_participant;

        return MoELayeredExpertOwnership(
            static_cast<int>(participants_.size()),
            std::move(owners),
            first_model_layer);
    }

} // namespace llaminar2
