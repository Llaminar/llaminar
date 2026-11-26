/**
 * @file MoEOverlayResidencyAuthorityInternal.h
 * @brief Shared internal economy scoring and swap-search helpers for MoEOverlayResidencyAuthority shards.
 */
#pragma once
/**
 * @file MoEOverlayResidencyAuthority.cpp
 * @brief Transactional implementation of heterogeneous MoE tier residency.
 *
 * The authority turns immutable routing histograms into capacity-preserving,
 * economy-gated migration cycles. Background transports stage weights before
 * a single epoch publication; ticket leases keep the old epoch alive until no
 * inference can reference it. Candidate construction also preserves physical
 * owners for experts that remain in a tier, so bounded waves never invent
 * unrelated participant-to-participant transfers.
 * Placement forecasts remain private predictions. Published transactions keep
 * their immutable observed routing window, including real batch boundaries;
 * smoothing cannot rewrite those observations or their movement metadata.
 * Jointly admitted seed cohorts and ordinary candidates share the same
 * fairness and capacity classification. Fairness is not a dependency edge;
 * only the exact candidate/cohort economy gate can authorize coupled work.
 * Publication and quiescence are distinct: retired banks and aborted transfers
 * retain their event-poll obligation until advanceBackground returns Idle.
 * Request admission observes the one-shot economy activation atomically; an
 * active request never joins the background candidate-scoring mutex.
 * Each authority owns one exact global layer interval. Plans and movement
 * records keep global identities; price, forecast and hysteresis arrays store
 * only that interval's compact rows, rejecting evidence from another stage.
 * Optional pre-cut evidence projects already-scored tier alternatives without
 * treating isolated gains as a jointly feasible migration backlog.
 * Cycle decomposition derives proposal-local exact-target adjacency once;
 * original edge order still owns deterministic DFS closure and publication.
 */

#include "MoEOverlayResidencyAuthority.h"
#include "MoEOverlayPreparedTargetService.h"
#include "MoEOptimizationMovementTopology.h"
#include "MoEExpertOverlayPreparationPlan.h"

#include "MoEOverlayEconomyCalibrationPlanner.h"
#include "MoEOverlayDistributedResidencyProtocol.h"
#include "MoEOverlayTransactionCost.h"
#include "MoEOverlayPreparedTransactionCost.h"

#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>


namespace llaminar2
{
    namespace {
        /**
         * @brief Express exact whole-owner targets relative to one canonical tier projection.
         * @param baseline Complete source-bound tier candidate, including original CPU cohorts.
         * @param desired Canonical destination rows retained by selected physical cycles.
         * @return Whole-owner corrections only; cohorts keep their original control blocks.
         * @throws std::logic_error For foreign rows, tier changes or altered cohort identity.
         * The canonical owner-map constructor validates physical endpoints and counts.
         */
        std::vector<MoELayeredExpertOwnershipChange> wholeOwnerTargetChanges(
            const MoEExpertOwnerMap &baseline, std::span<const MoEExpertOwner> desired)
        {
            std::vector<MoELayeredExpertOwnershipChange> result;
            for (const auto &target : desired)
            {
                const auto *source = baseline.placementFor(target.layer_idx, target.expert_id);
                if (!source || source->tier_idx != target.tier_idx)
                    throw std::logic_error("Whole participant correction must retain its canonical tier");
                if (source->shard_cohort || target.shard_cohort)
                {
                    if (!(*source == target) || source->shard_cohort.get() != target.shard_cohort.get() ||
                        source->shard_cohort.owner_before(target.shard_cohort) ||
                        target.shard_cohort.owner_before(source->shard_cohort))
                        throw std::logic_error("Whole participant correction cannot replace an original CPU cohort");
                    continue;
                }
                if (source->owner_participant != target.owner_participant)
                    result.push_back({.layer_idx = target.layer_idx, .expert_id = target.expert_id,
                        .previous_participant = source->owner_participant,
                        .current_participant = target.owner_participant});
            }
            return result;
        }

        /** @return Whether the original routed tiers contain complete CPU/GPU shard groups. */
        bool jointFragmentPlan(const MoERoutedExpertPlacementPlan& plan)
        {
            return MoEExpertOwnerMap::usesJointFragments(plan);
        }
        /** @brief Authenticate the original typed map against this exact model/plan once at cold construction. */
        void requireJointSource(const MoERoutedExpertPlacementPlan& plan,const MoERoutedExpertModelMetadata& model,
            const MoEExpertOverlayPreparationPlan& preparation)
        {
            const auto& original=preparation.jointOwnerMap();
            if(!original || !jointFragmentPlan(plan) || model.num_layers<=0 || model.num_experts<=0 ||
               plan.placements.size()!=size_t(model.num_layers) ||
               original->participants()!=MoEExpertOwnerMap::declaredParticipants(plan,{.reject_tensor_sharded_domains=false}) ||
               original->owners().size()!=size_t(model.num_layers)*size_t(model.num_experts))
                throw std::invalid_argument("Static joint residency requires its original complete source map and members");
            for(const auto& domain:plan.domains)
                if(domain.routed_compute_policy==RoutedExpertComputePolicy::TensorSharded && !domain.participants.empty() &&
                   (domain.participants.size()>1 || domain.participants.front().isGPU()) &&
                   (domain.scope != (domain.participants.front().isCPU() ? ExecutionDomainScope::NODE_LOCAL : ExecutionDomainScope::RANK_LOCAL) ||
                    domain.routed_phase_policy!=RoutedExpertPhasePolicy::Uniform ||
                    domain.routed_decode_assignment_policy!=RoutedExpertAssignmentPolicy::StaticOwner ||
                    domain.routed_prefill_assignment_policy!=RoutedExpertAssignmentPolicy::StaticOwner))
                    throw std::invalid_argument("Joint residency requires original domain membership and Uniform StaticOwner phases");
            for(size_t index=0;index<original->owners().size();++index) {
                const auto& row=original->owners()[index];
                const auto placement=std::find_if(plan.placements.begin(),plan.placements.end(),[&](const auto& layer){return layer.layer==row.layer_idx;});
                if(row.layer_idx!=model.first_model_layer+int(index/size_t(model.num_experts)) ||
                   row.expert_id!=int(index%size_t(model.num_experts)) || placement==plan.placements.end() ||
                   placement->routed_expert_tier.size()!=size_t(model.num_experts) ||
                   row.tier_idx!=placement->routed_expert_tier[size_t(row.expert_id)] || row.tier_idx<0 ||
                   size_t(row.tier_idx)>=plan.routed_tiers.size() ||
                   row.domain_name!=plan.routed_tiers[size_t(row.tier_idx)].domain ||
                   row.tier_name!=plan.routed_tiers[size_t(row.tier_idx)].name || !row.resident)
                    throw std::invalid_argument("Static joint residency source has foreign layer/expert/tier coordinates");
                if(row.shard_cohort) {
                    if(row.owner_participant!=-1 || row.device.is_valid() ||
                       row.shard_cohort->source().layer_idx!=row.layer_idx ||
                       row.shard_cohort->ownership(0).geometry().experts!=model.num_experts)
                        throw std::invalid_argument("Joint fragment source cannot publish a whole-expert owner alias");
                    for(const auto& member:row.shard_cohort->participants()) {
                        const auto* declared=original->participantForId(member.participant_id);
                        if(!declared || declared->world_rank!=member.world_rank || declared->address!=member.address ||
                           declared->device!=member.address.toLocalDeviceId() || declared->tier_idx!=row.tier_idx)
                            throw std::invalid_argument("Joint fragment source changed its original rank/device member");
                    }
                } else if(!original->participantForId(row.owner_participant)) {
                    throw std::invalid_argument("Joint snapshot contains an invalid ordinary whole-expert row");
                }
            }
        }
    }

    namespace
    {
        bool isDynamicPolicy(RoutedExpertResidencyPolicy policy)
        {
            return policy == RoutedExpertResidencyPolicy::HistogramTieredCache ||
                   policy == RoutedExpertResidencyPolicy::RoutedTierRebalanced;
        }

        int tierPriority(
            const MoERoutedExpertPlacementPlan &plan,
            int tier_idx)
        {
            if (tier_idx < 0 ||
                tier_idx >= static_cast<int>(plan.routed_tiers.size()))
            {
                throw std::logic_error(
                    "MoE overlay migration references an invalid tier index");
            }
            return plan.routed_tiers[static_cast<size_t>(tier_idx)].priority;
        }

        const char *directionName(MoEOverlayTierMigrationDirection direction)
        {
            switch (direction)
            {
            case MoEOverlayTierMigrationDirection::Promotion:
                return "promotion";
            case MoEOverlayTierMigrationDirection::Demotion:
                return "demotion";
            case MoEOverlayTierMigrationDirection::SamePriority:
                return "same_priority";
            }
            return "unknown";
        }

        /** @return Stable diagnostic spelling for one typed movement axis. */
        const char *movementAxisName(MoEOptimizationMovementAxis axis)
        {
            switch (axis)
            {
            case MoEOptimizationMovementAxis::TierResidency:
                return "tier_residency";
            case MoEOptimizationMovementAxis::ParticipantPlacement:
                return "participant_placement";
            case MoEOptimizationMovementAxis::Combined:
                return "combined";
            }
            return "unknown";
        }

        /** @return The placement objective(s) represented by @p cycle. */
        MoEOptimizationMovementAxis migrationCycleAxis(
            const MoEOverlayResidencyTransaction &transaction,
            const MoEOverlayTierMigrationCycle &cycle)
        {
            bool advances_tier = false;
            bool advances_participant = false;
            for (const std::size_t migration_index : cycle.migration_indices)
            {
                const auto &migration =
                    transaction.migrations.at(migration_index);
                advances_tier |= advancesTierResidency(migration.axis);
                advances_participant |=
                    advancesParticipantPlacement(migration.axis);
            }
            if (advances_tier && advances_participant)
                return MoEOptimizationMovementAxis::Combined;
            return advances_tier
                       ? MoEOptimizationMovementAxis::TierResidency
                       : MoEOptimizationMovementAxis::ParticipantPlacement;
        }

        /** @brief Compact PerfStats accounting for independent cycle axes. */
        struct MigrationCycleAxisCounts
        {
            std::size_t tier_residency = 0;
            std::size_t participant_placement = 0;
            std::size_t combined = 0;

            /** @brief Account one classified cycle exactly once. */
            void add(MoEOptimizationMovementAxis axis) noexcept
            {
                /*
                 * These are exclusive telemetry coordinates.  Combined
                 * means that one physical cycle advances both objectives; it
                 * must not also inflate the two pure-axis buckets.  The
                 * advances* predicates above answer scheduling questions and
                 * therefore intentionally have inclusive semantics.
                 */
                switch (axis)
                {
                case MoEOptimizationMovementAxis::TierResidency:
                    ++tier_residency;
                    break;
                case MoEOptimizationMovementAxis::ParticipantPlacement:
                    ++participant_placement;
                    break;
                case MoEOptimizationMovementAxis::Combined:
                    ++combined;
                    break;
                }
            }

            /** @return Number of cycles classified across exclusive axes. */
            [[nodiscard]] std::size_t total() const noexcept
            {
                return tier_residency + participant_placement + combined;
            }
        };

        /**
         * @brief One fairness reservation, satisfied only by admitted work.
         *
         * This is not a tier dependency. The candidate/cohort economy checks
         * prove dependencies before admission. A rejected tier candidate does
         * not forbid an independent participant improvement, and a seeded
         * dependency cohort can already satisfy the same participant lane.
         */
        class ParticipantLaneReservation
        {
        public:
            /** Result of observing one economically and physically admitted cycle. */
            enum class Transition { Unchanged, Fulfilled };

            /** @brief Construct an unneeded reservation for a single-axis wave. */
            constexpr ParticipantLaneReservation() noexcept = default;

            /** @return An outstanding one-lane reservation for a two-axis wave. */
            static constexpr ParticipantLaneReservation required() noexcept
            {
                ParticipantLaneReservation result;
                result.state_ = State::Pending;
                return result;
            }

            /**
             * @brief Observe actual admission, including a pre-admitted cohort.
             * @param axis Logical objectives of one admitted candidate cycle.
             * @return Fulfilled exactly once, when participant work is admitted.
             */
            constexpr Transition recordAdmission(MoEOptimizationMovementAxis axis) noexcept
            {
                if (state_ != State::Pending || !advancesParticipantPlacement(axis))
                    return Transition::Unchanged;
                state_ = State::Fulfilled;
                return Transition::Fulfilled;
            }

            /** @return Whether this wave requested independent-axis fairness. */
            constexpr bool active() const noexcept { return state_ != State::Unneeded; }
            /** @return Whether admitted work already satisfies the reservation. */
            constexpr bool fulfilled() const noexcept { return state_ == State::Fulfilled; }
            /** @return Zero or one reserved participant candidate, never a cycle total. */
            constexpr std::size_t reservedCandidates() const noexcept { return fulfilled() ? 1u : 0u; }

        private:
            /** Only admission may advance Pending to Fulfilled. */
            enum class State { Unneeded, Pending, Fulfilled };
            State state_ = State::Unneeded;
        };

        using CapacityKey = std::pair<int, int>;

        std::map<CapacityKey, int> tierCapacities(
            const MoERoutedExpertPlacementPlan &plan)
        {
            std::map<CapacityKey, int> capacities;
            for (const auto &placement : plan.placements)
            {
                for (const int tier_idx : placement.routed_expert_tier)
                    ++capacities[{placement.layer, tier_idx}];
            }
            return capacities;
        }

        /** @return Physical slots at every actual member, derived from the canonical map. */
        std::map<CapacityKey, int> participantCapacities(
            const MoEExpertOwnerMap &owner_map)
        {
            std::map<CapacityKey, int> capacities;
            for (const auto &owner : owner_map.owners())
            {
                const auto target = MoEExpertPlacementTarget::fromPlacement(owner);
                if (!target)
                    throw std::logic_error("Residency capacity requires a complete placement target");
                for (std::size_t member = 0; member < target->participantCount(); ++member)
                    ++capacities[{owner.layer_idx, target->participantId(member)}];
            }
            return capacities;
        }

        /** @brief Per-layer/tier evidence produced by participant balancing. */
        struct ParticipantRebalanceLayerEvidence
        {
            int layer_idx = -1;
            int tier_idx = -1;
            bool service_objective_used = false;
            uint64_t routed_window_activations = 0;
            uint64_t load_total = 0;
            uint64_t minimum_window_activations = 0;
            uint64_t load_min_before = 0;
            uint64_t load_max_before = 0;
            uint64_t load_min_after = 0;
            uint64_t load_max_after = 0;
            uint32_t swap_pairs = 0;
        };

        /** @brief Complete explicit owner table and its deterministic evidence. */
        struct ParticipantRebalancePlan
        {
            MoEExpertOwnerMap owner_map;
            std::vector<ParticipantRebalanceLayerEvidence> evidence;
            uint64_t swap_pairs = 0;

            [[nodiscard]] uint64_t ownerChanges() const noexcept
            {
                return swap_pairs * 2u;
            }
        };

        /**
         * @brief Select one host-authority paired ownership swap.
         *
         * This is the policy path for an authority without certified endpoint
         * service evidence. It deliberately delegates to the same exact raw
         * activation selector used by CUDA and ROCm device authorities. The
         * shared selector has no fixed participant limit when transfer-slot
         * masks are absent, so arbitrary NodeTP domains do not inherit the
         * device ABI's eight-participant storage bound.
         */
        moe_rebalance_policy::OwnershipSwapChoice
        bestRawOverlayParticipantSwap(
            const std::vector<uint64_t> &participant_load,
            const std::vector<uint64_t> &expert_counts,
            const std::vector<int32_t> &expert_owner,
            uint64_t routed_window_activations,
            const MoEOverlayParticipantRebalancePolicy &policy)
        {
            if (participant_load.size() < 2 ||
                expert_counts.empty() ||
                expert_counts.size() != expert_owner.size() ||
                participant_load.size() >
                    std::numeric_limits<uint32_t>::max() ||
                expert_counts.size() >
                    std::numeric_limits<uint32_t>::max())
            {
                return {};
            }
            return moe_rebalance_policy::bestDynamicOwnershipSwap(
                participant_load.data(),
                expert_counts.data(),
                expert_owner.data(),
                static_cast<uint32_t>(expert_counts.size()),
                static_cast<uint32_t>(participant_load.size()),
                moe_rebalance_policy::DynamicOwnershipEvidenceWindow{
                    .routed_activations = routed_window_activations,
                    .minimum_routed_activations =
                        policy.minimum_window_activations,
                },
                policy.imbalance_threshold_per_mille,
                policy.minimum_improvement_per_mille);
        }

        class ObservedTransactionServiceObjective;

        /**
         * @brief Select a paired owner swap using certified endpoint service.
         * @param participant_load Routed demand for each tier-local participant.
         * @param tier_expert_ids Original model expert IDs owned by this tier.
         * @param expert_counts Forecast demand for those expert IDs.
         * @param expert_owner Tier-local participant index for each expert.
         * @param tier_participant_ids Canonical physical IDs for the local indices.
         * @param complete_targets One complete whole/cohort target per model expert.
         * @param layer_idx Original model-global layer being evaluated.
         * @param routed_window_activations Observed window size for the evidence floor.
         * @param service_objective Checked phase service and complete target-price rows.
         * @param policy Required imbalance, improvement and evidence thresholds.
         * @return A paired swap, or an empty choice when no admissible gain exists.
         * @throws std::logic_error For incomplete targets or inconsistent owner indices.
         *
         * The selector uses the final gate's phase-specific service arithmetic
         * inside the tier: no recurring phase may regress, and the sum of the
         * tier's phase makespans must fall by the configured minimum fraction.
         * The later transaction gate remains authoritative for the complete
         * topology. Keeping the local search broader is necessary when a free
         * participant objective is absorbed into a tier-transfer cycle or two
         * jointly useful axes reduce the global critical path only together.
         * The function is defined after the checked wide-cost helpers below.
         */
        moe_rebalance_policy::OwnershipSwapChoice
        bestServiceAwareOverlayParticipantSwap(
            const std::vector<uint64_t> &participant_load,
            const std::vector<int> &tier_expert_ids,
            const std::vector<uint64_t> &expert_counts,
            const std::vector<int32_t> &expert_owner,
            const std::vector<int> &tier_participant_ids,
            std::span<const MoEOverlayRouteTarget> complete_targets,
            int layer_idx,
            uint64_t routed_window_activations,
            ObservedTransactionServiceObjective &service_objective,
            const MoEOverlayParticipantRebalancePolicy &policy);

        /** @brief Project canonical whole/cohort owners to their complete economy target. */
        MoEOverlayRouteTarget economyExecutionTarget(const MoEExpertOwner &owner);

        /**
         * @brief Balance apportioned owners inside every unchanged tier.
         * @param plan Complete tier-placement candidate before participant balancing.
         * @param base_owner_map Canonical physical owners for that candidate.
         * @param original_joint_owners Original CPU cohort/cut authority, or null for whole ownership.
         * @param window Checked routing demand with original model-layer coordinates.
         * @param policy Participant balancing thresholds and search limits.
         * @param service_objective Optional measured phase costs for complete targets.
         * @return An unpublished canonical owner map and deterministic swap evidence.
         * @throws std::logic_error For missing tiers, owners or complete service targets.
         * @throws std::invalid_argument If explicit changes violate original source or slot capacity.
         *
         * The input owner map already reflects the tier optimizer's candidate.
         * This pass changes only exact participants within each tier and only
         * through paired swaps, so logical tier quotas and physical live-slot
         * counts remain invariant. Replicated and tensor-sharded domains have
         * different residency semantics and are intentionally left unchanged.
         */
        ParticipantRebalancePlan planOverlayParticipantRebalance(
            const MoERoutedExpertPlacementPlan &plan,
            const MoEExpertOwnerMap &base_owner_map,
            const MoEExpertOwnerMap *original_joint_owners,
            const ValidatedDecodeExpertHistogramWindowView &window,
            const MoEOverlayParticipantRebalancePolicy &policy,
            std::optional<std::reference_wrapper<
                ObservedTransactionServiceObjective>> service_objective)
        {
            ParticipantRebalancePlan result{
                .owner_map = base_owner_map,
            };
            if (!policy.enabled)
                return result;
            std::vector<MoELayeredExpertOwnershipChange> whole_owner_changes;

            /*
             * Discover alternatives independently of publication capacity.
             * `maximum_plan_entries_per_wave` is an admission limit for the
             * immutable transaction assembled below, not a layer-order search
             * limit.  Applying it here made the first few skewed layers consume
             * the complete candidate budget before measured service economics
             * could compare them with later layers.  The transaction composer
             * owns the one authoritative entry and physical-cycle budget.
             */
            for (const auto &placement : plan.placements)
            {
                uint64_t routed_window_activations = 0u;
                for (int expert_id = 0;
                     expert_id < window.numExperts();
                     ++expert_id)
                {
                    const uint64_t count = window.activationCount(
                        placement.layer, expert_id);
                    routed_window_activations =
                        std::numeric_limits<uint64_t>::max() -
                                    routed_window_activations <
                                count
                            ? std::numeric_limits<uint64_t>::max()
                            : routed_window_activations + count;
                }
                // Service scratch contains complete execution targets for this
                // layer only. CPU cohort tags remain fixed; accepted whole-owner
                // pairs update their two targets before the next search round.
                // Keeping the vector across tiers also preserves earlier tier
                // corrections whenever complete transaction scoring reads them.
                std::vector<MoEOverlayRouteTarget> layer_targets;
                if (service_objective)
                {
                    layer_targets.reserve(static_cast<std::size_t>(window.numExperts()));
                    for (int expert = 0; expert < window.numExperts(); ++expert)
                    {
                        const auto *owner = base_owner_map.placementFor(placement.layer, expert);
                        if (!owner) throw std::logic_error("Participant service search lost a canonical layer target");
                        layer_targets.push_back(economyExecutionTarget(*owner));
                    }
                }
                uint32_t layer_swap_pairs = 0;
                for (std::size_t tier_idx = 0;
                     tier_idx < plan.routed_tiers.size();
                     ++tier_idx)
                {
                    if (layer_swap_pairs >= policy.maximum_swaps_per_layer)
                    {
                        break;
                    }

                    const auto &tier = plan.routed_tiers[tier_idx];
                    const auto domain_it = std::find_if(
                        plan.domains.begin(),
                        plan.domains.end(),
                        [&](const auto &domain)
                        { return domain.name == tier.domain; });
                    if (domain_it == plan.domains.end())
                    {
                        throw std::logic_error(
                            "ExpertOverlay participant planner lost a tier domain");
                    }
                    if (domain_it->routed_compute_policy !=
                        RoutedExpertComputePolicy::Apportioned)
                    {
                        continue;
                    }

                    const auto participant_ids =
                        base_owner_map.participantIdsForTier(
                            static_cast<int>(tier_idx));
                    if (participant_ids.size() < 2)
                        continue;

                    std::vector<int> expert_ids;
                    std::vector<uint64_t> expert_counts;
                    std::vector<int32_t> local_owners;
                    std::vector<uint64_t> participant_load(
                        participant_ids.size(), 0u);
                    for (int expert_id = 0;
                         expert_id < window.numExperts();
                         ++expert_id)
                    {
                        if (placement.routed_expert_tier[
                                static_cast<std::size_t>(expert_id)] !=
                            static_cast<int>(tier_idx))
                        {
                            continue;
                        }
                        const auto *placement_owner = result.owner_map.placementFor(placement.layer, expert_id);
                        if (!placement_owner || placement_owner->shard_cohort)
                            throw std::logic_error("Participant search requires one canonical independently movable owner");
                        const int owner = placement_owner->owner_participant;
                        const auto participant_it = std::find(
                            participant_ids.begin(),
                            participant_ids.end(),
                            owner);
                        if (participant_it == participant_ids.end())
                        {
                            throw std::logic_error(
                                "ExpertOverlay candidate owner is outside its selected tier");
                        }
                        const auto local_owner = static_cast<int32_t>(
                            std::distance(
                                participant_ids.begin(), participant_it));
                        const uint64_t count = window.activationCount(
                            placement.layer, expert_id);
                        expert_ids.push_back(expert_id);
                        expert_counts.push_back(count);
                        local_owners.push_back(local_owner);
                        participant_load[
                            static_cast<std::size_t>(local_owner)] += count;
                    }
                    if (expert_ids.size() < participant_ids.size())
                        continue;

                    const auto [before_min_it, before_max_it] =
                        std::minmax_element(
                            participant_load.begin(),
                            participant_load.end());
                    uint64_t load_total = 0u;
                    for (const uint64_t load : participant_load)
                    {
                        load_total =
                            std::numeric_limits<uint64_t>::max() -
                                        load_total <
                                    load
                                ? std::numeric_limits<uint64_t>::max()
                                : load_total + load;
                    }
                    ParticipantRebalanceLayerEvidence evidence{
                        .layer_idx = placement.layer,
                        .tier_idx = static_cast<int>(tier_idx),
                        .service_objective_used =
                            service_objective.has_value(),
                        .routed_window_activations =
                            routed_window_activations,
                        .load_total = load_total,
                        .minimum_window_activations =
                            policy.minimum_window_activations,
                        .load_min_before = *before_min_it,
                        .load_max_before = *before_max_it,
                        .load_min_after = *before_min_it,
                        .load_max_after = *before_max_it,
                    };

                    while (layer_swap_pairs <
                           policy.maximum_swaps_per_layer)
                    {
                        const auto choice = service_objective
                            ? bestServiceAwareOverlayParticipantSwap(
                                  participant_load,
                                  expert_ids,
                                  expert_counts,
                                  local_owners,
                                  participant_ids,
                                  layer_targets,
                                  placement.layer,
                                  routed_window_activations,
                                  service_objective->get(),
                                  policy)
                            : bestRawOverlayParticipantSwap(
                                  participant_load,
                                  expert_counts,
                                  local_owners,
                                  routed_window_activations,
                                  policy);
                        if (!choice.valid)
                            break;
                        if (!moe_rebalance_policy::applyDynamicOwnershipSwap(
                                participant_load.data(),
                                local_owners.data(),
                                choice))
                        {
                            throw std::logic_error(
                                "ExpertOverlay participant swap could not update planner scratch");
                        }
                        /*
                         * Service-aware selection scores every participant in
                         * the layer, not only this tier's scratch slice. Make
                         * the accepted scratch exchange visible immediately so
                         * a second greedy swap is evaluated from the exact
                         * provisional ownership state it would publish.
                         */
                        if (service_objective)
                        {
                            layer_targets.at(static_cast<std::size_t>(expert_ids.at(choice.heavy_expert))) =
                                MoEOverlayRouteTarget::whole(participant_ids.at(static_cast<std::size_t>(
                                    local_owners.at(choice.heavy_expert))));
                            layer_targets.at(static_cast<std::size_t>(expert_ids.at(choice.light_expert))) =
                                MoEOverlayRouteTarget::whole(participant_ids.at(static_cast<std::size_t>(
                                    local_owners.at(choice.light_expert))));
                        }
                        ++layer_swap_pairs;
                        ++result.swap_pairs;
                        ++evidence.swap_pairs;
                    }

                    const auto [after_min_it, after_max_it] =
                        std::minmax_element(
                            participant_load.begin(),
                            participant_load.end());
                    evidence.load_min_after = *after_min_it;
                    evidence.load_max_after = *after_max_it;
                    result.evidence.push_back(evidence);
                    // The existing tier-local swap scratch authors only final
                    // whole-owner corrections. Canonical source validation and
                    // physical count checks run once below, after every tier.
                    for (std::size_t index = 0; index < expert_ids.size(); ++index)
                    {
                        const auto *source = base_owner_map.placementFor(placement.layer, expert_ids[index]);
                        const int destination = participant_ids.at(static_cast<std::size_t>(local_owners[index]));
                        if (source->owner_participant != destination)
                            whole_owner_changes.push_back({.layer_idx = placement.layer,
                                .expert_id = expert_ids[index], .previous_participant = source->owner_participant,
                                .current_participant = destination});
                    }
                }
            }
            if (!whole_owner_changes.empty())
            {
                if (original_joint_owners)
                    result.owner_map = MoEExpertOwnerMap::buildShardTransition(
                        plan, base_owner_map, *original_joint_owners, whole_owner_changes);
                else
                {
                    auto ownership = base_owner_map.layeredOwnership(
                        window.numLayers(), window.numExperts(), window.firstModelLayer());
                    for (const auto &change : whole_owner_changes)
                        ownership.assignOwner(change.layer_idx, change.expert_id, change.current_participant);
                    result.owner_map = MoEExpertOwnerMap::buildExplicit(plan, ownership);
                }
            }
            return result;
        }

        PerfStatsCollector::Tags baseTags(
            RoutedExpertResidencyPolicy policy,
            uint64_t epoch)
        {
            return {
                {"policy", toString(policy)},
                {"epoch", std::to_string(epoch)},
            };
        }

        /** @brief Retained production phases in service-profile column order. */
        constexpr std::array<ExpertHistogramSource,
                             kExpertHistogramProductionSourceCount>
            kProductionHistogramSources{
                ExpertHistogramSource::DecodeToken,
                ExpertHistogramSource::PrefillChunk,
                ExpertHistogramSource::GroupedVerifier,
            };

        using WideCost = unsigned __int128;

        /** @brief Convert an exact wide non-negative cost to the public ABI. */
        uint64_t checkedCostToU64(
            WideCost value,
            const char *description)
        {
            if (value > std::numeric_limits<uint64_t>::max())
            {
                throw std::overflow_error(
                    std::string("ExpertOverlay ") + description +
                    " exceeds the uint64 evidence range");
            }
            return static_cast<uint64_t>(value);
        }

        /** @brief Add a cost without permitting silent wide-integer wrap. */
        void checkedAddWide(
            WideCost *total,
            WideCost increment,
            const char *description)
        {
            constexpr WideCost kMaximum = ~WideCost{0};
            if (!total || increment > kMaximum - *total)
            {
                throw std::overflow_error(
                    std::string("ExpertOverlay ") + description +
                    " exceeds the exact wide-cost range");
            }
            *total += increment;
        }

        /** @brief Multiply exact cost terms without permitting silent wrap. */
        WideCost checkedMultiplyWide(
            WideCost lhs,
            WideCost rhs,
            const char *description)
        {
            constexpr WideCost kMaximum = ~WideCost{0};
            if (lhs != 0 && rhs > kMaximum / lhs)
            {
                throw std::overflow_error(
                    std::string("ExpertOverlay ") + description +
                    " exceeds the exact wide-cost range");
            }
            return lhs * rhs;
        }


        /** @param owner Original canonical row. @return Its complete logical execution target. */
        MoEOverlayRouteTarget economyExecutionTarget(const MoEExpertOwner& owner) {
            if(!MoEExpertPlacementTarget::fromPlacement(owner))
                throw std::invalid_argument("Economy source has no checked complete target");
            return owner.shard_cohort?MoEOverlayRouteTarget::cohort(owner.tier_idx):
                MoEOverlayRouteTarget::whole(owner.owner_participant);
        }
        /**
         * @brief One proposal's borrowed observations and reusable service scratch.
         *
         * Candidate ranking may smooth counts, but service is priced only against
         * immutable real invocations. This adapter borrows the authority's already
         * certified participant/layer table; it does not create a second profile.
         * It is exclusive to the proposal lock, not inference or a device controller.
         */
        class ObservedTransactionServiceObjective
        {
        public:
            /** @brief Phase-pure critical paths and selected-tier imbalance evidence. */
            struct Score
            {
                std::array<moe_overlay_economy::ServiceCostPair,
                           kExpertHistogramProductionSourceCount> by_phase{};
                std::array<WideCost, kExpertHistogramProductionSourceCount>
                    minimum_before_by_phase{};
            };

            /**
             * @brief One fixed-layout candidate search retaining invocation boundaries.
             *
             * Owning each transaction's derived inputs prevents a later accepted
             * swap or calibration publication from changing a search in progress.
             * The next search is constructed from the newly accepted owner map.
             */
            class SwapSearch
            {
            public:
                /**
                 * @brief Price a two-expert exchange in every observed phase.
                 * @param first First logical expert ID in the complete model.
                 * @param second Second logical expert ID in the complete model.
                 * @return Exact sums of independent transaction critical paths.
                 */
                Score scoreSwap(std::uint32_t first, std::uint32_t second) const
                {
                    Score result;
                    for (std::size_t phase = 0; phase < transactions_.size(); ++phase)
                        for (const auto &transaction : transactions_[phase])
                        {
                            requireComplete(moe_overlay_economy::appendTransaction(
                                result.by_phase[phase], moe_overlay_economy::repeatPreparedTransaction(
                                    transaction.cost.scoreSwap(first, second), transaction.occurrences)));
                            checkedAddWide(&result.minimum_before_by_phase[phase],
                                checkedMultiplyWide(transaction.cost.minimumBefore(), transaction.occurrences,
                                                    "swap source imbalance floor"),
                                "swap source imbalance floor");
                        }
                    return result;
                }

            private:
                friend class ObservedTransactionServiceObjective;
                /** @brief Equal phase-local transactions retain exact multiplicity. */
                struct Transaction
                {
                    moe_overlay_economy::PreparedTransactionSwapSearch cost;
                    std::uint64_t occurrences;
                };
                std::array<std::vector<Transaction>, kExpertHistogramProductionSourceCount> transactions_;
            };

            /**
             * @brief Borrow one authenticated observation and the certified table.
             * @param observed Real routing sample retained by the proposal.
             * @param topology Reachable and economy-priced phases.
             * @param rows Participant-major, layer-minor certified price pointers.
             * @param participants Complete physical participant count.
             *
             * Scratch is constructed once per proposal, not per candidate or batch.
             * A counts-only window is rejected if and when service is requested;
             * forecast maintenance alone does not pretend to quote execution cost.
             */
            ObservedTransactionServiceObjective(
                ValidatedDecodeExpertHistogramWindowView observed,
                const ExpertHistogramProductionTopology &topology,
                std::span<const MoERoutedTargetLayerPhaseServiceCost *const> rows,
                std::span<const MoEOverlayRouteTarget> targets)
                : observed_(observed), topology_(topology), rows_(rows),targets_(targets),
                  prices_(targets.size()), scratch_(targets.size())
            {
                const auto participants=targets.size();
                if (participants == 0 ||
                    participants > std::numeric_limits<std::uint32_t>::max() ||
                    topology.layerCount() != static_cast<std::size_t>(observed.numLayers()) ||
                    topology.firstModelLayer() != observed.firstModelLayer() ||
                    rows.size() != participants * topology.layerCount())
                    throw std::logic_error(
                        "ExpertOverlay observed service objective has invalid certified geometry");
            }

            /** @param owner Original checked owner. @return Internal logical vertex, never a synthetic participant ID. */
            int32_t targetIndex(const MoEExpertOwner& owner) const {
                return targetIndex(economyExecutionTarget(owner));
            }

            /** @param target Canonical complete execution target, never a physical cohort member.
             * @return Its dense certified price-row index, independent of participant numbering. */
            int32_t targetIndex(MoEOverlayRouteTarget target) const {
                const auto found=std::lower_bound(targets_.begin(),targets_.end(),target.word(),
                    [](auto value,int word){return value.word()<word;});
                if(found==targets_.end() || found->word()!=target.word())
                    throw std::invalid_argument("Observed economy lacks complete target service evidence");
                return int32_t(found-targets_.begin());
            }

            /** @return Authenticated expert geometry, never inferred from a candidate. */
            int numExperts() const noexcept { return observed_.numExperts(); }

            /** @return Exact observed demand, not smoothed candidate-ranking counts. */
            std::uint64_t activationCount(int layer, int expert) const
            {
                return observed_.activationCount(layer, expert);
            }

            /**
             * @brief Prepare exact source work once for all candidate swaps.
             * @param layer Routed layer whose immutable observations are priced.
             * @param owners Complete source map for this single search round.
             * @param selected Participants of the tier being rebalanced.
             * @return Self-contained search with no live-map or price-table aliases.
             *
             * General multi-move admission still uses score(). This narrower
             * operation is only for the exhaustive two-owner exchange search;
             * it does not approximate combined movements or pool token batches.
             */
            SwapSearch prepareSwapSearch(int layer, std::span<const std::int32_t> owners,
                                         std::span<const int> selected)
            {
                const auto &prepared = prepareLayer(layer);
                SwapSearch result;
                for (std::size_t phase = 0; phase < kProductionHistogramSources.size(); ++phase)
                {
                    if (prepared.transactions[phase].empty()) continue;
                    if (!topology_.reachable(layer, phase))
                        throw std::logic_error("ExpertOverlay swap found demand in an unreachable phase");
                    if (!topology_.requiresServiceEvidence(layer, phase)) continue;
                    for (std::size_t participant = 0; participant < prices_.size(); ++participant)
                        prices_[participant] = rows_[
                            participant * topology_.layerCount() + storageIndexForModelLayer(layer)]
                            ->nanoseconds_per_activation[phase];
                    auto &transactions = result.transactions_[phase];
                    transactions.reserve(prepared.transactions[phase].size());
                    for (const auto &[batch, occurrences] : prepared.transactions[phase])
                        transactions.push_back({
                            moe_overlay_economy::PreparedTransactionSwapSearch(batch, owners, prices_, selected),
                            occurrences});
                }
                return result;
            }

            /**
             * @brief Price complete before/after maps at the observed invocation boundaries.
             * @param layer Exact routed layer.
             * @param before Published or provisional complete owner map for this layer.
             * @param after Candidate complete owner map for this layer.
             * @param selected Empty means the complete topology; otherwise price only
             *        the named tier's critical path for the intentionally broader search.
             * @return Sum of per-invocation maxima, separately for each recurring phase.
             * @throws On missing observations, invalid owners, unpriced work or overflow.
             *
             * Selecting a tier does not regroup transactions. Its imbalance floor is
             * likewise the sum of per-invocation minima, not a minimum of window totals.
             */
            Score score(int layer, std::span<const std::int32_t> before,
                        std::span<const std::int32_t> after,
                        std::span<const int> selected = {})
            {
                if (before.size() != static_cast<std::size_t>(numExperts()) ||
                    after.size() != before.size())
                    throw std::logic_error("ExpertOverlay observed service owner geometry mismatch");
                for (std::size_t index = 0; index < before.size(); ++index)
                {
                    if (before[index] < 0 || after[index] < 0 ||
                        static_cast<std::size_t>(before[index]) >= scratch_.size() ||
                        static_cast<std::size_t>(after[index]) >= scratch_.size())
                        throw std::logic_error("ExpertOverlay observed service owner is invalid");
                }
                for (std::size_t index = 0; index < selected.size(); ++index)
                {
                    if (selected[index] < 0 ||
                        static_cast<std::size_t>(selected[index]) >= scratch_.size() ||
                        std::find(selected.begin(), selected.begin() + index,
                                  selected[index]) != selected.begin() + index)
                        throw std::logic_error("ExpertOverlay observed service tier is invalid");
                }

                const auto &prepared = prepareLayer(layer);
                Score result;
                for (std::size_t phase = 0; phase < kProductionHistogramSources.size(); ++phase)
                {
                    for (std::size_t participant = 0; participant < prices_.size(); ++participant)
                        prices_[participant] = rows_[
                            participant * topology_.layerCount() + storageIndexForModelLayer(layer)]
                            ->nanoseconds_per_activation[phase];
                    for (const auto &[batch, occurrences] : prepared.transactions[phase])
                    {
                        if (!topology_.reachable(layer, phase))
                            throw std::logic_error(
                                "ExpertOverlay observed service found demand in an unreachable phase");
                        // Rare retained catch-up/tail execution can be reachable
                        // without belonging to the declared recurring payoff model.
                        if (!topology_.requiresServiceEvidence(layer, phase))
                            continue;
                        auto transaction = batch.score(
                            {before.data(), after.data(), prices_.data(),
                             static_cast<std::uint32_t>(numExperts()),
                             static_cast<std::uint32_t>(prices_.size())},
                            scratch_.data(), static_cast<std::uint32_t>(scratch_.size()));
                        requireComplete(transaction.status);
                        std::uint64_t minimum = std::numeric_limits<std::uint64_t>::max();
                        if (selected.empty())
                        {
                            for (const auto &work : scratch_)
                                minimum = std::min(minimum, work.before_ns);
                        }
                        else
                        {
                            transaction.cost = {};
                            for (const int participant : selected)
                            {
                                const auto &work = scratch_[static_cast<std::size_t>(participant)];
                                transaction.cost.before_ns =
                                    std::max(transaction.cost.before_ns, work.before_ns);
                                transaction.cost.after_ns =
                                    std::max(transaction.cost.after_ns, work.after_ns);
                                minimum = std::min(minimum, work.before_ns);
                            }
                        }
                        // Identical invocation histograms have identical costs
                        // for every candidate. Multiply each already-reduced
                        // critical path, never pool independent token rows.
                        requireComplete(moe_overlay_economy::appendTransaction(
                            result.by_phase[phase], moe_overlay_economy::repeatPreparedTransaction(
                                transaction, occurrences)));
                        checkedAddWide(&result.minimum_before_by_phase[phase],
                                       checkedMultiplyWide(minimum, occurrences,
                                                           "repeated observed imbalance floor"),
                                       "observed transaction imbalance floor");
                    }
                }
                return result;
            }

        private:
            /**
             * @brief Translate an owned global identity into the certified price row.
             * @param layer Model-global routed layer, never an allocation index.
             * @return Compact row shared by observation and price topology.
             * @throws std::out_of_range If another stage's identity is supplied.
             */
            std::size_t storageIndexForModelLayer(int layer) const
            {
                if (layer < observed_.firstModelLayer() ||
                    layer - observed_.firstModelLayer() >= observed_.numLayers())
                    throw std::out_of_range("ExpertOverlay service layer is outside its stage");
                return static_cast<std::size_t>(layer - observed_.firstModelLayer());
            }

            /** @brief Phase-pure equivalent invocations and exact multiplicities. */
            struct PreparedLayer
            {
                int layer;
                std::array<std::map<moe_overlay_economy::PreparedTransactionCost, std::uint64_t>,
                           kExpertHistogramProductionSourceCount> transactions;
            };

            /**
             * @brief Compact routing once per layer, not once per candidate swap.
             * @param layer Immutable observation's routed layer index.
             * @return Equivalent invocation groups retaining every batch's cost.
             *
             * Keep only the active layer: a long decode window must not multiply
             * scratch by the model's layer count. Neither owners nor prices are
             * cached, because accepted provisional swaps change the former.
             */
            const PreparedLayer &prepareLayer(int layer)
            {
                if (prepared_layer_ && prepared_layer_->layer == layer)
                    return *prepared_layer_;
                const auto &demand = observed_.transactionDemand();
                const auto batches = demand.layerTransactions(layer);
                PreparedLayer prepared{.layer = layer, .transactions = {}};
                for (std::size_t batch = 0; batch < batches.size(); ++batch)
                {
                    const auto source = std::find(kProductionHistogramSources.begin(),
                        kProductionHistogramSources.end(), batches[batch].phase);
                    if (source == kProductionHistogramSources.end())
                        throw std::logic_error("ExpertOverlay transaction has an unknown production phase");
                    auto &groups = prepared.transactions[static_cast<std::size_t>(
                        source - kProductionHistogramSources.begin())];
                    auto [entry, inserted] = groups.try_emplace(
                        moe_overlay_economy::PreparedTransactionCost(
                            demand.routes(layer, batch), static_cast<std::uint32_t>(numExperts())), 0);
                    // At most batches.size() observations exist. A size_t
                    // indexed window cannot overflow this uint64 count.
                    ++entry->second;
                }
                prepared_layer_ = std::move(prepared);
                return *prepared_layer_;
            }

            /** @brief A partial or saturated cost is never eligible for publication. */
            static void requireComplete(moe_overlay_economy::TransactionCostStatus status)
            {
                using Status = moe_overlay_economy::TransactionCostStatus;
                if (status == Status::Complete) return;
                if (status == Status::Overflow)
                    throw std::overflow_error("ExpertOverlay observed transaction service overflows");
                throw std::logic_error(
                    "ExpertOverlay observed transaction service failed: status=" +
                    std::to_string(static_cast<std::uint32_t>(status)));
            }

            ValidatedDecodeExpertHistogramWindowView observed_;
            const ExpertHistogramProductionTopology &topology_;
            std::span<const MoERoutedTargetLayerPhaseServiceCost *const> rows_;
            std::span<const MoEOverlayRouteTarget> targets_;
            std::vector<std::uint64_t> prices_;
            std::vector<moe_overlay_economy::ServiceCostPair> scratch_;
            std::optional<PreparedLayer> prepared_layer_;
        };

        /**
         * @brief Project migration sets onto one published layer and price their delta.
         *
         * Both bounded candidate selection and final admission use this projection.
         * Each set is relative to the same immutable source map; combining axes must
         * not apply a move twice or price one expert as if two owners executed it.
         */
        ObservedTransactionServiceObjective::Score scoreObservedMigrationSet(
            ObservedTransactionServiceObjective &objective,
            const MoEOverlayResidencyTransaction &transaction,
            int layer, std::span<const std::size_t> before_indices,
            std::span<const std::size_t> after_indices)
        {
            std::vector<std::int32_t> before(objective.numExperts(), -1);
            // The immutable original map already owns canonical layer/expert
            // order. Read the exact layer range, retaining cohort targets and
            // the original checked expert-ID writes, not a parallel owner index.
            for (const auto &owner : transaction.previous->placementOwnerMap().placementsForLayer(layer))
                before.at(static_cast<std::size_t>(owner.expert_id)) = objective.targetIndex(owner);
            auto after = before;
            const auto apply = [&](std::span<const std::size_t> indices,
                                   std::vector<std::int32_t> &owners)
            {
                for (const auto index : indices)
                {
                    const auto &move = transaction.migrations.at(index);
                    if (move.layer_idx != layer) continue;
                    auto &owner = owners.at(static_cast<std::size_t>(move.expert_id));
                    if (owner != objective.targetIndex(move.source))
                        throw std::logic_error(
                            "ExpertOverlay economy migration source disagrees with published ownership");
                    owner = objective.targetIndex(move.destination);
                }
            };
            apply(before_indices, before);
            apply(after_indices, after);
            return objective.score(layer, before, after);
        }

        moe_rebalance_policy::OwnershipSwapChoice
        bestServiceAwareOverlayParticipantSwap(
            const std::vector<uint64_t> &participant_load,
            const std::vector<int> &tier_expert_ids,
            const std::vector<uint64_t> &expert_counts,
            const std::vector<int32_t> &expert_owner,
            const std::vector<int> &tier_participant_ids,
            std::span<const MoEOverlayRouteTarget> complete_targets,
            int layer_idx,
            uint64_t routed_window_activations,
            ObservedTransactionServiceObjective &service_objective,
            const MoEOverlayParticipantRebalancePolicy &policy)
        {
            moe_rebalance_policy::OwnershipSwapChoice choice{};
            const std::size_t candidate_count = tier_expert_ids.size();
            if (candidate_count < 2u ||
                expert_counts.size() != candidate_count ||
                expert_owner.size() != candidate_count ||
                tier_participant_ids.size() != participant_load.size() ||
                tier_participant_ids.size() < 2u ||
                routed_window_activations <
                    policy.minimum_window_activations)
            {
                return choice;
            }
            std::vector<std::int32_t> before_owners(service_objective.numExperts());
            if (complete_targets.size() != before_owners.size())
                throw std::logic_error("Participant service search requires one complete typed target per expert");
            for (int expert = 0; expert < service_objective.numExperts(); ++expert)
                before_owners[static_cast<std::size_t>(expert)] =
                    service_objective.targetIndex(complete_targets[static_cast<std::size_t>(expert)]);
            std::vector<int> selected_targets;
            selected_targets.reserve(tier_participant_ids.size());
            for (const int participant : tier_participant_ids)
                selected_targets.push_back(service_objective.targetIndex(MoEOverlayRouteTarget::whole(participant)));
            const auto baseline = service_objective.score(
                layer_idx, before_owners, before_owners, selected_targets);
            WideCost before_total = 0;
            bool imbalance_threshold_met = false;
            for (std::size_t phase = 0; phase < kProductionHistogramSources.size(); ++phase)
            {
                const WideCost tier_max = baseline.by_phase[phase].before_ns;
                const WideCost tier_min = baseline.minimum_before_by_phase[phase];
                checkedAddWide(&before_total, tier_max, "observed participant service");
                if (tier_max != 0 &&
                    (tier_min == 0 ||
                     checkedMultiplyWide(tier_max, 1000u, "observed imbalance numerator") >=
                     checkedMultiplyWide(tier_min, policy.imbalance_threshold_per_mille,
                                         "observed imbalance denominator")))
                    imbalance_threshold_met = true;
            }
            if (!imbalance_threshold_met || before_total == 0)
                return choice;

            const auto swap_search = service_objective.prepareSwapSearch(
                layer_idx, before_owners, selected_targets);
            WideCost best_after_total = ~WideCost{0};
            for (std::size_t first = 0;
                 first < candidate_count;
                 ++first)
            {
                const int32_t first_local_owner = expert_owner[first];
                if (first_local_owner < 0 ||
                    first_local_owner >=
                        static_cast<int32_t>(tier_participant_ids.size()))
                {
                    throw std::logic_error(
                        "ExpertOverlay service-aware candidate has an invalid tier-local owner");
                }
                for (std::size_t second = first + 1u;
                     second < candidate_count;
                     ++second)
                {
                    const int32_t second_local_owner = expert_owner[second];
                    if (second_local_owner < 0 ||
                        second_local_owner >= static_cast<int32_t>(
                            tier_participant_ids.size()))
                    {
                        throw std::logic_error(
                            "ExpertOverlay service-aware candidate has an invalid tier-local owner");
                    }
                    if (first_local_owner == second_local_owner)
                        continue;

                    // Keep forecast counts for deterministic candidate ordering and
                    // raw-load bookkeeping, but never use them as parallel work.
                    const auto first_expert = static_cast<std::size_t>(tier_expert_ids[first]);
                    const auto second_expert = static_cast<std::size_t>(tier_expert_ids[second]);
                    // Swapping two unobserved experts leaves every transaction
                    // unchanged. Forecast marginals are not this proof: use
                    // the authenticated current window, independently of any
                    // historical smoothing used to rank candidates.
                    if (service_objective.activationCount(layer_idx, static_cast<int>(first_expert)) == 0 &&
                        service_objective.activationCount(layer_idx, static_cast<int>(second_expert)) == 0)
                        continue;
                    const auto measured = swap_search.scoreSwap(
                        static_cast<std::uint32_t>(first_expert),
                        static_cast<std::uint32_t>(second_expert));
                    WideCost after_total = 0;
                    bool phase_regressed = false;
                    for (std::size_t phase = 0; phase < kProductionHistogramSources.size(); ++phase)
                    {
                        const auto &cost = measured.by_phase[phase];
                        phase_regressed |= cost.after_ns > cost.before_ns;
                        checkedAddWide(&after_total, cost.after_ns,
                                       "observed candidate participant service");
                    }
                    if (phase_regressed || after_total >= before_total)
                        continue;

                    const WideCost improvement =
                        before_total - after_total;
                    if (policy.minimum_improvement_per_mille != 0u &&
                        checkedMultiplyWide(
                            improvement,
                            1000u,
                            "service-aware improvement numerator") <
                            checkedMultiplyWide(
                                before_total,
                                policy.minimum_improvement_per_mille,
                                "service-aware improvement denominator"))
                    {
                        continue;
                    }

                    const bool better =
                        !choice.valid ||
                        after_total < best_after_total ||
                        (after_total == best_after_total &&
                         (tier_expert_ids[first] <
                              tier_expert_ids[choice.heavy_expert] ||
                          (tier_expert_ids[first] ==
                               tier_expert_ids[choice.heavy_expert] &&
                           tier_expert_ids[second] <
                               tier_expert_ids[choice.light_expert])));
                    if (!better)
                        continue;

                    if (participant_load[static_cast<std::size_t>(
                                             first_local_owner)] <
                            expert_counts[first] ||
                        participant_load[static_cast<std::size_t>(
                                             second_local_owner)] <
                            expert_counts[second])
                    {
                        throw std::logic_error(
                            "ExpertOverlay service-aware raw load disagrees with candidate ownership");
                    }
                    const uint64_t first_after_load =
                        participant_load[static_cast<std::size_t>(
                            first_local_owner)] -
                        expert_counts[first] + expert_counts[second];
                    const uint64_t second_after_load =
                        participant_load[static_cast<std::size_t>(
                            second_local_owner)] -
                        expert_counts[second] + expert_counts[first];
                    uint64_t new_min = std::numeric_limits<uint64_t>::max();
                    uint64_t new_max = 0u;
                    for (std::size_t participant = 0;
                         participant < participant_load.size();
                         ++participant)
                    {
                        uint64_t load = participant_load[participant];
                        if (participant == static_cast<std::size_t>(
                                               first_local_owner))
                        {
                            load = first_after_load;
                        }
                        else if (participant == static_cast<std::size_t>(
                                                    second_local_owner))
                        {
                            load = second_after_load;
                        }
                        new_min = std::min(new_min, load);
                        new_max = std::max(new_max, load);
                    }

                    choice.overloaded_participant =
                        static_cast<uint32_t>(first_local_owner);
                    choice.underloaded_participant =
                        static_cast<uint32_t>(second_local_owner);
                    choice.heavy_expert = static_cast<uint32_t>(first);
                    choice.light_expert = static_cast<uint32_t>(second);
                    choice.heavy_count = expert_counts[first];
                    choice.light_count = expert_counts[second];
                    /*
                     * applyDynamicOwnershipSwap treats these legacy names as
                     * the exact pre-swap endpoint loads. Service-aware choices
                     * need not move the numerically heaviest expert away from
                     * the endpoint with the largest raw activation total.
                     */
                    choice.old_max_load = participant_load[
                        static_cast<std::size_t>(first_local_owner)];
                    choice.old_min_load = participant_load[
                        static_cast<std::size_t>(second_local_owner)];
                    choice.new_min_load = new_min;
                    choice.new_max_load = new_max;
                    choice.improvement = checkedCostToU64(
                        improvement,
                        "service-aware participant improvement");
                    choice.valid = true;
                    best_after_total = after_total;
                }
            }
            return choice;
        }

        /** @brief Multiply allocation geometry before constructing vectors. */
        std::size_t checkedSizeProduct(
            std::size_t lhs,
            std::size_t rhs,
            const char *description)
        {
            if (lhs != 0 &&
                rhs > std::numeric_limits<std::size_t>::max() / lhs)
            {
                throw std::overflow_error(
                    std::string("ExpertOverlay ") + description +
                    " overflows size_t");
            }
            return lhs * rhs;
        }

        /** @brief Nearest-integer deterministic smoothing with wide products. */
        uint64_t smoothCount(
            uint64_t previous,
            uint64_t current,
            const MoEOverlayMigrationEconomyPolicy &policy)
        {
            const uint64_t total_weight =
                static_cast<uint64_t>(policy.historical_window_weight) +
                static_cast<uint64_t>(policy.current_window_weight);
            const WideCost numerator =
                static_cast<unsigned __int128>(previous) *
                    policy.historical_window_weight +
                static_cast<unsigned __int128>(current) *
                    policy.current_window_weight +
                total_weight / 2u;
            const WideCost smoothed = numerator / total_weight;
            if (smoothed > std::numeric_limits<uint64_t>::max())
            {
                throw std::overflow_error(
                    "ExpertOverlay smoothed histogram count exceeds uint64");
            }
            return static_cast<uint64_t>(smoothed);
        }

        /**
         * @brief Private placement prediction paired with its last observed input.
         *
         * Only candidate ranking borrows the rounded counts. The actual input
         * remains immutable and is the sole source of executable activity and
         * transaction geometry. Keeping both in one value makes advancing the
         * prediction atomic even if validation, allocation or arithmetic fails.
         */
        class PlacementDemandForecast final
        {
        public:
            /**
             * @brief Seed a prediction without claiming observed transactions.
             * @param observed Complete immutable input retained for generation identity.
             * @throws std::invalid_argument If the input is absent or malformed.
             */
            explicit PlacementDemandForecast(
                std::shared_ptr<const DecodeExpertHistogramWindow> observed)
                : observed_(std::move(observed))
            {
                if (!observed_ || !observed_->valid())
                    throw std::invalid_argument("ExpertOverlay forecast requires valid observed demand");
                counts_ = *observed_;
                // Rounded marginals describe no particular collection of
                // batches. Never attach the observed trace to that prediction.
                counts_.transaction_demand.reset();
            }

            /**
             * @brief Advance once per observed generation, with strong exception safety.
             * @param observed New complete input, or an identical retransmission.
             * @param policy Validated deterministic history/current weights.
             * @throws std::invalid_argument On stale, changed or malformed input.
             * @throws std::overflow_error If rounded phase sums exceed their ABI.
             */
            void observe(std::shared_ptr<const DecodeExpertHistogramWindow> observed,
                         const MoEOverlayMigrationEconomyPolicy &policy)
            {
                if (!observed || !observed->valid() || !policy.valid() ||
                    observed->num_layers != counts_.num_layers ||
                    observed->first_model_layer != counts_.first_model_layer ||
                    observed->num_experts != counts_.num_experts ||
                    observed->generation < observed_->generation)
                    throw std::invalid_argument("ExpertOverlay forecast received invalid or regressing demand");
                if (observed->generation == observed_->generation)
                {
                    // The canonical wire identity includes batch boundaries
                    // and co-occurrence. Equal marginals alone are not equal
                    // evidence, nor is a separately owned identical copy new.
                    if (fingerprintDecodeExpertHistogramWindow(*observed) !=
                        fingerprintDecodeExpertHistogramWindow(*observed_))
                        throw std::invalid_argument(
                            "ExpertOverlay migration economy received different evidence for one histogram generation");
                    return;
                }

                PlacementDemandForecast next(std::move(observed));
                WideCost token_sum = 0;
                for (std::size_t phase = 0; phase < kExpertHistogramProductionSourceCount; ++phase)
                {
                    next.counts_.source_token_counts[phase] = smoothCount(
                        counts_.source_token_counts[phase], next.counts_.source_token_counts[phase], policy);
                    token_sum += next.counts_.source_token_counts[phase];
                }
                // Round each primitive once, then derive its redundant total.
                // Rounding the aggregate independently can disagree by a row.
                next.counts_.token_count = checkedCostToU64(token_sum, "forecast token total");
                for (std::size_t index = 0; index < counts_.source_expert_counts.size(); ++index)
                    next.counts_.source_expert_counts[index] = smoothCount(
                        counts_.source_expert_counts[index], next.counts_.source_expert_counts[index], policy);
                const std::size_t entries = counts_.expert_counts.size();
                for (std::size_t entry = 0; entry < entries; ++entry)
                {
                    WideCost phase_sum = 0;
                    for (std::size_t phase = 0; phase < kExpertHistogramProductionSourceCount; ++phase)
                        phase_sum += next.counts_.source_expert_counts[phase * entries + entry];
                    next.counts_.expert_counts[entry] = checkedCostToU64(phase_sum, "forecast expert total");
                }
                if (!next.counts_.valid())
                    throw std::logic_error("ExpertOverlay forecast produced invalid candidate counts");
                *this = std::move(next);
            }

            /** @return Whether this exact owned immutable input remains forecast-authoritative. */
            [[nodiscard]] bool observes(const std::shared_ptr<const DecodeExpertHistogramWindow> &window) const noexcept
            {
                return observed_.get() == window.get() &&
                    !observed_.owner_before(window) && !window.owner_before(observed_);
            }

            /** @return Borrowed prediction for candidate ranking, never publication. */
            [[nodiscard]] const DecodeExpertHistogramWindow &candidateCounts() const noexcept
            {
                return counts_;
            }

            /** @return Canonical count identity for the root's private policy audit. */
            [[nodiscard]] uint64_t fingerprint() const
            {
                return fingerprintDecodeExpertHistogramWindow(counts_);
            }

        private:
            std::shared_ptr<const DecodeExpertHistogramWindow> observed_;
            DecodeExpertHistogramWindow counts_;
        };
    } // namespace

    namespace
    {
        /**
         * @brief Resolve complete immutable targets without inventing a scalar cohort owner.
         * @param migration Original logical expert edge and both physical placements.
         * @return Source/destination targets, or no value for foreign/ambiguous coordinates.
         */
        std::optional<std::pair<MoEExpertPlacementTarget, MoEExpertPlacementTarget>>
        migrationPlacementTargets(const MoEOverlayTierMigration &migration) noexcept
        {
            if (migration.layer_idx < 0 || migration.expert_id < 0 ||
                migration.source.layer_idx != migration.layer_idx ||
                migration.destination.layer_idx != migration.layer_idx ||
                migration.source.expert_id != migration.expert_id ||
                migration.destination.expert_id != migration.expert_id)
                return std::nullopt;
            auto source = MoEExpertPlacementTarget::fromPlacement(migration.source);
            auto destination = MoEExpertPlacementTarget::fromPlacement(migration.destination);
            if (!source || !destination)
                return std::nullopt;
            if (migration.source.shard_cohort && migration.destination.shard_cohort)
            {
                const auto &from = *migration.source.shard_cohort;
                const auto &to = *migration.destination.shard_cohort;
                // Different admitted cuts may redistribute the same original
                // matrix. A different full matrix/format is not redistribution.
                if (from.source() != to.source() ||
                    from.ownership(0).geometry() != to.ownership(0).geometry())
                    return std::nullopt;
            }
            return std::pair{std::move(*source), std::move(*destination)};
        }
    }

    /**
     * @brief Reference-counted publication record for one immutable epoch.
     *
     * The shared pointer protects the record itself while `active_tickets`
     * protects the prepared engines named by its snapshot. The two lifetimes
     * are intentionally distinct: a racing ticket acquisition may briefly hold
     * this record without ever becoming an inference ticket.
     */
    struct MoEOverlayResidencyAuthority::PublishedEpochState
    {
        /** @brief Admission state for complete graph-sequence readers. */
        enum class GraphSequenceAdmissionState : std::uint8_t
        {
            Open, ///< New graph sequences may pin this current epoch.
            PublicationReserved, ///< A prepared successor owns the next boundary.
            Superseded, ///< Publication completed; retry against the successor.
        };

        explicit PublishedEpochState(
            std::shared_ptr<const MoEOverlayResidencySnapshot> published_snapshot)
            : snapshot(std::move(published_snapshot))
        {
        }

        std::shared_ptr<const MoEOverlayResidencySnapshot> snapshot;
        std::atomic<uint64_t> active_tickets{0};
        /** Complete graph sequences currently executing against this epoch. */
        std::atomic<uint64_t> active_graph_sequences{0};
        /**
         * Lock-free gate preventing selector publication inside a graph sequence.
         *
         * A publisher closes this gate only after every heavyweight preparation
         * edge is ready. Readers which already won admission continue without
         * interruption; later readers wait for the successor epoch rather than
         * combining an old host descriptor with a new device-bank selector.
         */
        std::atomic<GraphSequenceAdmissionState> graph_sequence_admission{
            GraphSequenceAdmissionState::Open};
        /**
         * Retirement closes exact-epoch admission before its final zero-count
         * observation.  A racing acquirer increments first and then rechecks
         * this flag, making reclamation safe without an inference-side mutex.
         */
        std::atomic<bool> accepting_exact_tickets{true};
    };

    /** @brief Maintenance-worker state for the single unpublished candidate. */
    struct MoEOverlayResidencyAuthority::ActiveBackgroundWave
    {
        enum class Phase
        {
            Staging,
            Preparing,
            AwaitingGraphSequenceBoundary,
            Publishing,
        };

        MoEOverlayResidencyTransaction transaction;
        std::shared_ptr<PublishedEpochState> previous;
        /** Same object exposed to exact tickets before selector fan-out. */
        std::shared_ptr<PublishedEpochState> candidate;
        std::unique_ptr<IMoEOverlayResidencyWave> work;
        Phase phase = Phase::Staging;
        /** Count a draining boundary once even if the worker polls repeatedly. */
        bool graph_sequence_drain_recorded = false;
    };

    /** @brief Published old epoch waiting for its final inference lease to drain. */
    struct MoEOverlayResidencyAuthority::PendingRetirement
    {
        /** @brief Typed process-local RCU grace-period progression. */
        enum class LocalGracePeriodState : std::uint8_t
        {
            AdmissionOpen = 0, ///< Old captured tickets may still claim E.
            AdmissionClosed,     ///< No new E ticket can enter; readers drain only.
        };

        std::shared_ptr<PublishedEpochState> previous;
        std::unique_ptr<IMoEOverlayResidencyWave> work;
        LocalGracePeriodState local_grace_period =
            LocalGracePeriodState::AdmissionOpen;
    };

    /** @brief Unpublished wave retained until all abort event edges quiesce. */
    struct MoEOverlayResidencyAuthority::PendingAbort
    {
        uint64_t previous_epoch = 0;
        std::unique_ptr<IMoEOverlayResidencyWave> work;
    };

    /**
     * @brief Maintenance-owned deterministic history and validated cost tables.
     *
     * Profile pointers borrow immutable shared objects retained by `Config`.
     * Flattened tables make every proposal lookup total and allocation-free
     * after construction. `last_moved_generation` changes only after an epoch
     * is published, so an aborted or deferred wave cannot consume hysteresis.
     */
    struct MoEOverlayResidencyAuthority::EconomyState
    {
        int first_model_layer = 0;
        int num_layers = 0;
        int num_experts = 0;
        int tier_count = 0;
        int participant_count = 0; ///< Number of complete execution targets, not fragment members.
        std::vector<MoEOverlayRouteTarget> execution_targets;
        ExpertHistogramProductionTopology production_topology;
        std::vector<const MoERoutedTargetLayerPhaseServiceCost *>
            participant_service_cost_rows; ///< Historical name: compact complete-target slots, never cohort members.
        std::vector<const MoEOverlayTargetLayerMigrationCost *>
            migration_cost_rows; ///< Dense source/destination complete-target slots, not physical participant IDs.
        std::optional<PlacementDemandForecast> forecast;
        std::vector<uint64_t> last_moved_generation;

        /** @param target Actual whole/cohort word. @return Checked internal execution-vertex index. */
        size_t targetIndex(MoEOverlayRouteTarget target) const {
            const auto found=std::lower_bound(execution_targets.begin(),execution_targets.end(),target.word(),
                [](auto value,int word){return value.word()<word;});
            if(found==execution_targets.end() || found->word()!=target.word())
                throw std::invalid_argument("Economy target is outside original complete service declarations");
            return size_t(found-execution_targets.begin());
        }

        /**
         * @brief Authenticate a global layer before indexing stage-local evidence.
         * @param layer Global model identity retained by physical movement records.
         * @return Compact price and hysteresis row.
         * @throws std::out_of_range For a layer outside the admitted stage.
         */
        std::size_t storageIndexForModelLayer(int layer) const
        {
            if (layer < first_model_layer || layer - first_model_layer >= num_layers)
                throw std::out_of_range("ExpertOverlay economy layer is outside its stage");
            return static_cast<std::size_t>(layer - first_model_layer);
        }

        /**
         * @param source_participant Certified source endpoint ID.
         * @param destination_participant Distinct certified destination ID.
         * @param layer Owned global model layer.
         * @return Prevalidated directed endpoint/layer cost row.
         * @throws std::out_of_range If the coordinate exceeds admitted storage.
         */
        const MoEOverlayTargetLayerMigrationCost &migrationCost(
            MoEOverlayRouteTarget source_target,
            MoEOverlayRouteTarget destination_target,
            int layer) const
        {
            const std::size_t participants = execution_targets.size();
            return *migration_cost_rows.at(
                (targetIndex(source_target) * participants + targetIndex(destination_target)) *
                    static_cast<std::size_t>(num_layers) +
                storageIndexForModelLayer(layer));
        }

        /**
         * @param layer Owned global model layer.
         * @param expert Logical routed expert ID.
         * @return Compact hysteresis index, independent of the stage origin.
         * @throws std::out_of_range If either coordinate is outside the stage.
         */
        std::size_t expertOffset(int layer, int expert) const
        {
            if (expert < 0 || expert >= num_experts)
                throw std::out_of_range("ExpertOverlay economy expert is out of range");
            return storageIndexForModelLayer(layer) *
                       static_cast<std::size_t>(num_experts) +
                   static_cast<std::size_t>(expert);
        }
    };
} // namespace llaminar2
