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

        /** @return Whether the immutable plan contains actual CPU tensor-shard members. */
        bool jointCPUPlan(const MoERoutedExpertPlacementPlan& plan)
        {
            return std::any_of(plan.domains.begin(),plan.domains.end(),[](const auto& domain){
                return domain.routed_compute_policy==RoutedExpertComputePolicy::TensorSharded &&
                    domain.participants.size()>1 && std::any_of(domain.participants.begin(),domain.participants.end(),
                        [](const auto& member){return member.isCPU();});
            });
        }
        /** @brief Authenticate the original typed map against this exact model/plan once at cold construction. */
        void requireJointSource(const MoERoutedExpertPlacementPlan& plan,const MoERoutedExpertModelMetadata& model,
            const MoEExpertOverlayPreparationPlan& preparation)
        {
            const auto& original=preparation.cpuJointOwnerMap();
            if(!original || !jointCPUPlan(plan) || model.num_layers<=0 || model.num_experts<=0 ||
               plan.placements.size()!=size_t(model.num_layers) ||
               original->participants()!=MoEExpertOwnerMap::declaredParticipants(plan,{.reject_tensor_sharded_domains=false}) ||
               original->owners().size()!=size_t(model.num_layers)*size_t(model.num_experts))
                throw std::invalid_argument("Static joint residency requires its original complete source map and members");
            for(const auto& domain:plan.domains)
                if(domain.routed_compute_policy==RoutedExpertComputePolicy::TensorSharded &&
                   std::any_of(domain.participants.begin(),domain.participants.end(),[](const auto& member){return member.isCPU();}) &&
                   (domain.scope!=ExecutionDomainScope::NODE_LOCAL || domain.routed_phase_policy!=RoutedExpertPhasePolicy::Uniform ||
                    domain.routed_decode_assignment_policy!=RoutedExpertAssignmentPolicy::StaticOwner ||
                    domain.routed_prefill_assignment_policy!=RoutedExpertAssignmentPolicy::StaticOwner))
                    throw std::invalid_argument("CPU joint residency requires original node-local Uniform StaticOwner phases");
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
                           declared->device!=DeviceId::cpu() || declared->tier_idx!=row.tier_idx)
                            throw std::invalid_argument("Joint fragment source changed its original CPU rank/NUMA member");
                    }
                } else if(!original->participantForId(row.owner_participant)) {
                    throw std::invalid_argument("Joint snapshot contains an invalid ordinary whole-expert row");
                }
            }
        }
    }

    MoEOverlayResidencyOwnershipKind MoEOverlayResidencySnapshot::ownershipKind() const noexcept
    {
        return joint_ ? MoEOverlayResidencyOwnershipKind::CPUJointFragments
                      : MoEOverlayResidencyOwnershipKind::WholeExperts;
    }

    std::shared_ptr<const MoEExpertOwnerMap> MoEOverlayResidencySnapshot::jointOwnerMap() const noexcept
    {
        return joint_ ? joint_->live_owners : nullptr;
    }

    std::shared_ptr<const MoEExpertOverlayPreparationPlan>
    MoEOverlayResidencySnapshot::originalProjectionPreparation() const noexcept
    {
        return joint_ ? joint_->preparation : nullptr;
    }

    const MoEExpertOwnerMap &MoEOverlayResidencySnapshot::placementOwnerMap() const
    {
        if (!joint_)
            return owner_map;
        if (!joint_->live_owners)
            throw std::logic_error("Joint snapshot lost its canonical live placement");
        return *joint_->live_owners;
    }

    int MoEOverlayResidencySnapshot::firstModelLayer() const noexcept
    {
        return joint_ ? (joint_->source_plan ? joint_->source_plan->first_model_layer : 0)
                      : layered_ownership.firstModelLayer();
    }

    int MoEOverlayResidencySnapshot::layerCount() const noexcept
    {
        return joint_ ? (joint_->source_plan ? int(joint_->source_plan->placements.size()) : 0)
                      : layered_ownership.layerCount();
    }

    int MoEOverlayResidencySnapshot::expertCount() const noexcept
    {
        return joint_ ? (joint_->source_plan && !joint_->source_plan->placements.empty()
                            ? int(joint_->source_plan->placements.front().routed_expert_tier.size())
                            : 0)
                      : layered_ownership.expertCount();
    }

    const MoEExpertOwnerMap& MoEOverlayResidencySnapshot::originalOwnerMap() const
    {
        if (!joint_)
            return owner_map;
        const auto original = joint_->preparation->cpuJointOwnerMap();
        if (!original)
            throw std::logic_error("Joint snapshot lost its original source map");
        return *original;
    }

    bool MoEOverlayTierMigration::crossesWorldRank() const noexcept
    {
        if (!source.shard_cohort && !destination.shard_cohort)
            return source.owner_world_rank_known && destination.owner_world_rank_known &&
                source.owner_world_rank != destination.owner_world_rank;
        if (source.shard_cohort && destination.shard_cohort)
        {
            for (std::size_t from = 0; from < source.shard_cohort->participants().size(); ++from)
                for (std::size_t to = 0; to < destination.shard_cohort->participants().size(); ++to)
                {
                    // All three CPU fragment projections use the SAME F
                    // interval (Gate/Up N and Down K). Empty intersections do
                    // not create remote movement, even if ranks differ.
                    const auto a = source.shard_cohort->ownership(from).projection(WeightRole::MoEExpertUp);
                    const auto b = destination.shard_cohort->ownership(to).projection(WeightRole::MoEExpertUp);
                    if (std::max(a.first_row, b.first_row) <
                            std::min(a.first_row + a.rows, b.first_row + b.rows) &&
                        source.shard_cohort->participants()[from].world_rank !=
                            destination.shard_cohort->participants()[to].world_rank)
                        return true;
                }
            return false;
        }
        const auto &cohort = source.shard_cohort ? source.shard_cohort : destination.shard_cohort;
        const auto &whole = source.shard_cohort ? destination : source;
        return whole.owner_world_rank_known && std::any_of(cohort->participants().begin(),
            cohort->participants().end(), [&](const auto &member) { return member.world_rank != whole.owner_world_rank; });
    }

    bool MoEOverlayResidencySnapshot::valid() const noexcept
    {
        if (!epoch || !placement_plan)
            return false;
        if (joint_)
            return epoch == joint_->epoch && joint_->source_plan && joint_->preparation &&
                   joint_->live_owners && !joint_->live_owners->owners().empty() &&
                   placement_plan.get() == joint_->live_plan.get() &&
                   !placement_plan.owner_before(joint_->live_plan) &&
                   !joint_->live_plan.owner_before(placement_plan) && layered_ownership.empty() &&
                   owner_map.owners().empty() && owner_map.participants().empty();
        return !layered_ownership.empty() &&
               placement_plan->first_model_layer == layered_ownership.firstModelLayer();
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
            return owner.shard_cohort?MoEOverlayRouteTarget::cpuCohort(owner.tier_idx):
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

    bool MoEOverlayMigrationEconomyPolicy::valid() const noexcept
    {
        return current_window_weight > 0 &&
               static_cast<uint64_t>(historical_window_weight) +
                       static_cast<uint64_t>(current_window_weight) <=
                   std::numeric_limits<uint32_t>::max() &&
               payoff_horizon_tokens > 0;
    }

    bool MoEOverlayMigrationEconomyEvidence::valid(
        bool has_migrations) const noexcept
    {
        if (!enabled)
        {
            return service_profile_identity.empty() &&
                   migration_profile_identity.empty() &&
                   smoothed_through_generation == 0 &&
                   forecast_fingerprint == 0 &&
                   historical_window_weight == 0 &&
                   current_window_weight == 0 &&
                   payoff_horizon_tokens == 0 &&
                   minimum_net_benefit_ns == 0 &&
                   minimum_residency_generations == 0 &&
                   projected_service_gain_ns == 0 &&
                   projected_transfer_and_repack_ns == 0 &&
                   projected_inference_interference_ns == 0 &&
                   projected_net_benefit_ns == 0 &&
                   payoff_rejected_cycles == 0 &&
                   residency_rejected_cycles == 0;
        }

        if (service_profile_identity.empty() ||
            migration_profile_identity.empty() ||
            forecast_fingerprint == 0 ||
            current_window_weight == 0 || payoff_horizon_tokens == 0 ||
            static_cast<uint64_t>(historical_window_weight) +
                    static_cast<uint64_t>(current_window_weight) >
                std::numeric_limits<uint32_t>::max())
        {
            return false;
        }
        if (!has_migrations)
        {
            return projected_service_gain_ns == 0 &&
                   projected_transfer_and_repack_ns == 0 &&
                   projected_inference_interference_ns == 0 &&
                   projected_net_benefit_ns == 0;
        }
        if (projected_transfer_and_repack_ns >
            std::numeric_limits<uint64_t>::max() -
                projected_inference_interference_ns)
        {
            return false;
        }
        const uint64_t measured_cost =
            projected_transfer_and_repack_ns +
            projected_inference_interference_ns;
        return projected_service_gain_ns >= measured_cost &&
               projected_net_benefit_ns ==
                   projected_service_gain_ns - measured_cost &&
               projected_net_benefit_ns > minimum_net_benefit_ns;
    }

    bool MoEOverlayAuthoritativeResidencyEntry::valid() const noexcept
    {
        const bool valid_axis =
            axis == MoEOptimizationMovementAxis::TierResidency ||
            axis == MoEOptimizationMovementAxis::ParticipantPlacement ||
            axis == MoEOptimizationMovementAxis::Combined;
        return candidate_tier_idx >= 0 &&
               !candidate_target.isAbsent() &&
               (!candidate_target.isCPUCohort() ||
                candidate_target.word() == -std::int64_t(candidate_tier_idx) - 2) && valid_axis &&
               (changed ||
                (activation_count == 0 && estimated_weight_bytes == 0));
    }

    bool MoEOverlayAuthoritativeResidencyPlan::valid() const noexcept
    {
        if (expected_epoch == 0 || num_layers <= 0 || num_experts <= 0 ||
            !histogram_window || !histogram_window->valid() ||
            histogram_window->num_layers != num_layers ||
            histogram_window->num_experts != num_experts)
        {
            return false;
        }
        const auto layers = static_cast<size_t>(num_layers);
        const auto experts = static_cast<size_t>(num_experts);
        if (layers > std::numeric_limits<size_t>::max() / experts ||
            entries.size() != layers * experts)
        {
            return false;
        }
        for (size_t index = 0; index < entries.size(); ++index)
        {
            const auto &entry = entries[index];
            if (!entry.valid())
                return false;

            /*
             * The dense entry and histogram are one authenticated semantic
             * value, not merely two independently well-formed payloads. A
             * follower reconstructs the migration economy from this exact
             * evidence, so accepting a different count here would postpone a
             * coordinator defect until after publication.
             */
            if (entry.changed &&
                entry.activation_count !=
                    histogram_window->expert_counts[index])
            {
                return false;
            }
        }
        return true;
    }

    size_t MoEOverlayAuthoritativeResidencyPlan::offset(
        int layer_idx,
        int expert_id) const
    {
        if (!histogram_window || !histogram_window->containsModelLayer(layer_idx) ||
            histogram_window->num_layers != num_layers || expert_id < 0 ||
            expert_id >= num_experts)
        {
            throw std::out_of_range(
                "ExpertOverlay authoritative plan coordinate is out of range");
        }
        return static_cast<size_t>(layer_idx - firstModelLayer()) *
                   static_cast<size_t>(num_experts) +
               static_cast<size_t>(expert_id);
    }

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

    bool MoEOverlayTierMigrationCycle::valid(
        const std::vector<MoEOverlayTierMigration> &migrations) const noexcept
    {
        if (layer_idx < 0 || migration_indices.empty())
            return false;
        // A repeated index necessarily repeats its logical (layer, expert)
        // key. The existing key set rejects both without clearing a full-wave
        // bitmap for every two-edge cycle in a large candidate population.
        std::set<std::pair<int, int>> logical_edges;
        for (size_t offset = 0; offset < migration_indices.size(); ++offset)
        {
            const size_t index = migration_indices[offset];
            const size_t next_index = migration_indices[(offset + 1) % migration_indices.size()];
            if (index >= migrations.size() || next_index >= migrations.size())
                return false;
            const auto &edge = migrations[index];
            const auto &next = migrations[next_index];
            if (!logical_edges.emplace(edge.layer_idx, edge.expert_id).second)
                return false;
            const auto targets = migrationPlacementTargets(edge);
            const auto next_targets = migrationPlacementTargets(next);
            if (edge.layer_idx != layer_idx || next.layer_idx != layer_idx ||
                !targets || !next_targets || targets->second != next_targets->first)
                return false;
        }
        return true;
    }

    MoEOverlayMigrationCapacityEvidence analyzeMoEOverlayMigrationCapacity(
        std::span<const MoEOverlayTierMigration> migrations) noexcept
    {
        using FlowKey = std::tuple<int, int>;
        std::map<FlowKey, std::int64_t> participant_flow;
        std::map<FlowKey, std::int64_t> tier_flow;
        std::set<FlowKey> logical_edges;
        MoEOverlayMigrationCapacityEvidence evidence;
        evidence.edges_checked = migrations.size();
        for (const auto &migration : migrations)
        {
            const auto targets = migrationPlacementTargets(migration);
            if (!targets || !logical_edges.emplace(migration.layer_idx, migration.expert_id).second)
            {
                ++evidence.malformed_edges;
                continue;
            }
            const auto &[source, destination] = *targets;
            // One logical edge moves every source/destination fragment. Physical
            // slots are per member; tier slots and demand are counted only once.
            for (size_t member = 0; member < source.participantCount(); ++member)
                --participant_flow[{migration.layer_idx, source.participantId(member)}];
            for (size_t member = 0; member < destination.participantCount(); ++member)
                ++participant_flow[{migration.layer_idx, destination.participantId(member)}];
            --tier_flow[{migration.layer_idx, source.tierIndex()}];
            ++tier_flow[{migration.layer_idx, destination.tierIndex()}];
        }
        evidence.participant_coordinates_checked = participant_flow.size();
        evidence.tier_coordinates_checked = tier_flow.size();
        evidence.participant_flow_violations = static_cast<size_t>(std::count_if(
            participant_flow.begin(), participant_flow.end(),
            [](const auto &entry) { return entry.second != 0; }));
        evidence.tier_flow_violations = static_cast<size_t>(std::count_if(
            tier_flow.begin(), tier_flow.end(),
            [](const auto &entry) { return entry.second != 0; }));
        return evidence;
    }

    bool MoEOverlayResidencyTransaction::valid() const noexcept
    {
        if (!previous || !candidate || !previous->valid() ||
            !candidate->valid() || expected_epoch != previous->epoch ||
            expected_epoch == std::numeric_limits<uint64_t>::max() ||
            candidate->epoch != expected_epoch + 1 ||
            previous->ownershipKind() != candidate->ownershipKind() ||
            previous->firstModelLayer() != candidate->firstModelLayer() ||
            previous->layerCount() != candidate->layerCount() ||
            previous->expertCount() != candidate->expertCount())
        {
            return false;
        }
        const bool joint = previous->ownershipKind() ==
                           MoEOverlayResidencyOwnershipKind::CPUJointFragments;
        const auto same_owner = [](const auto &a, const auto &b)
        { return a.get() == b.get() && !a.owner_before(b) && !b.owner_before(a); };
        const auto matches_placement = [&](const MoEExpertOwner &canonical,
                                           const MoEExpertOwner &edge)
        {
            if (joint)
                return canonical == edge && same_owner(canonical.shard_cohort, edge.shard_cohort);
            return canonical.owner_participant == edge.owner_participant &&
                   canonical.tier_idx == edge.tier_idx && canonical.device == edge.device;
        };
        if (joint &&
            (!same_owner(previous->joint_->source_plan, candidate->joint_->source_plan) ||
             !same_owner(previous->joint_->preparation, candidate->joint_->preparation)))
            return false;
        const bool calibration =
            purpose ==
            MoEOverlayResidencyTransactionPurpose::EconomyCalibration;
        const bool restoration =
            purpose ==
            MoEOverlayResidencyTransactionPurpose::PreparedContextRestoration;
        if ((!calibration && !restoration &&
             purpose != MoEOverlayResidencyTransactionPurpose::PlacementChange))
            return false;
        if (calibration)
        {
            if (joint && (migrations.size() != 2 || migration_cycles.size() != 1))
                return false; // This admitted calibration shape is one reciprocal target pair.
            /*
             * A calibration candidate is deliberately the unchanged live
             * topology at the next private epoch. Only its shadow copies are
             * exercised, and the transport makes commit unrepresentable.
             */
            if (calibration_sequence == 0 || histogram_window ||
                histogram_generation != 0 || host_admission ||
                economy.enabled ||
                !same_owner(previous->placement_plan, candidate->placement_plan) ||
                previous->layered_ownership != candidate->layered_ownership ||
                (joint && (!same_owner(previous->joint_->live_plan, candidate->joint_->live_plan) ||
                           !same_owner(previous->joint_->live_owners, candidate->joint_->live_owners))) ||
                (!joint && (previous->owner_map.owners() != candidate->owner_map.owners() ||
                            previous->owner_map.participants() != candidate->owner_map.participants())))
            {
                return false;
            }
        }
        else if (calibration_sequence != 0)
        {
            return false;
        }
        if (restoration &&
            (histogram_window || histogram_generation != 0 ||
             host_admission || economy.enabled))
        {
            return false;
        }
        if (histogram_window &&
            (!histogram_window->valid() ||
             histogram_window->generation != histogram_generation ||
             histogram_window->first_model_layer != previous->firstModelLayer() ||
             histogram_window->num_layers != previous->layerCount() ||
             histogram_window->num_experts != previous->expertCount()))
        {
            return false;
        }

        if (!economy.valid(!migrations.empty()))
            return false;

        if (host_admission &&
            (!host_admission->valid() || !candidate ||
             host_admission->transaction != candidate->epoch ||
             host_admission->candidate_epoch != candidate->epoch ||
             host_admission->admitted_physical_cycles !=
                 migration_cycles.size()))
        {
            return false;
        }

        const auto &previous_map = previous->placementOwnerMap();
        const auto &candidate_map = candidate->placementOwnerMap();
        if (joint && !calibration)
        {
            // Every changed logical placement must have one physical edge.
            // Otherwise an empty or partial wave could publish uncopied rows.
            std::size_t changed = 0;
            for (const auto &destination : candidate_map.owners())
            {
                const auto *source = previous_map.placementFor(destination.layer_idx,
                                                               destination.expert_id);
                if (!source)
                    return false;
                changed += !(*source == destination);
            }
            if (changed != migrations.size())
                return false;
        }
        if (migrations.empty())
        {
            return migration_cycles.empty() && shadow_requirements.empty() &&
                   !host_admission;
        }

        if (!analyzeMoEOverlayMigrationCapacity(migrations)
                 .capacityPreserved())
        {
            return false;
        }

        std::vector<size_t> coverage(migrations.size(), 0);
        for (const auto &cycle : migration_cycles)
        {
            if (!cycle.valid(migrations))
                return false;
            for (const size_t index : cycle.migration_indices)
                ++coverage[index];
        }
        if (std::any_of(
                coverage.begin(),
                coverage.end(),
                [](size_t count)
                { return count != 1; }))
        {
            return false;
        }

        using RequirementKey = std::tuple<int, int, int>;
        std::map<RequirementKey, size_t> expected_requirements;
        for (const auto &migration : migrations)
        {
            if (migration.layer_idx < 0 || migration.expert_id < 0 ||
                migration.source.layer_idx != migration.layer_idx ||
                migration.destination.layer_idx != migration.layer_idx ||
                migration.source.expert_id != migration.expert_id ||
                migration.destination.expert_id != migration.expert_id ||
                migration.source.tier_idx < 0 ||
                migration.destination.tier_idx < 0)
            {
                return false;
            }
            if (calibration && migration.activation_count != 0)
                return false; // Shadow timing never authors logical routing demand.
            if (histogram_window)
            {
                if (!histogram_window->containsModelLayer(migration.layer_idx) ||
                    migration.expert_id >= histogram_window->num_experts)
                {
                    return false;
                }
                const size_t histogram_index =
                    static_cast<size_t>(migration.layer_idx - histogram_window->first_model_layer) *
                        static_cast<size_t>(histogram_window->num_experts) +
                    static_cast<size_t>(migration.expert_id);
                if (migration.activation_count !=
                    histogram_window->expert_counts[histogram_index])
                {
                    return false;
                }
            }
            if (migration.axis ==
                    MoEOptimizationMovementAxis::ParticipantPlacement &&
                migration.crossesTier())
            {
                return false;
            }
            const auto *previous_owner = previous_map.placementFor(
                migration.layer_idx, migration.expert_id);
            if (!previous_owner || !matches_placement(*previous_owner, migration.source))
            {
                return false;
            }
            if (calibration)
            {
                if (joint)
                {
                    // Shadow destinations are complete original targets already
                    // present in this exact source layer. Borrow the canonical
                    // member/cut/control block, changing only the migrant expert
                    // coordinate; calibration never rewrites live ownership.
                    bool authenticated = false;
                    for (const auto &resident : previous_map.owners())
                    {
                        if (resident.layer_idx != migration.layer_idx) continue;
                        const auto canonical = MoEExpertPlacementTarget::fromPlacement(resident);
                        const auto actual = MoEExpertPlacementTarget::fromPlacement(migration.destination);
                        const auto &destination = migration.destination;
                        // Compare the actual target without copying strings or
                        // manufacturing a temporary owner inside noexcept validation.
                        if (canonical && actual && *canonical == *actual &&
                            same_owner(resident.shard_cohort, destination.shard_cohort) &&
                            resident.owner_participant == destination.owner_participant &&
                            resident.device == destination.device && resident.resident == destination.resident &&
                            resident.tier_name == destination.tier_name && resident.domain_name == destination.domain_name &&
                            resident.domain_participant_index == destination.domain_participant_index &&
                            resident.owner_world_rank == destination.owner_world_rank &&
                            resident.owner_world_rank_known == destination.owner_world_rank_known &&
                            resident.address == destination.address)
                        {
                            authenticated = true;
                            break;
                        }
                    }
                    if (!authenticated) return false;
                }
                else
                {
                    const auto *destination_participant = previous_map.participantForId(
                        migration.destination.owner_participant);
                    if (!destination_participant ||
                        destination_participant->tier_idx != migration.destination.tier_idx ||
                        destination_participant->device != migration.destination.device)
                        return false;
                }
            }
            else
            {
                const auto *candidate_owner = candidate_map.placementFor(
                    migration.layer_idx, migration.expert_id);
                if (!candidate_owner || !matches_placement(*candidate_owner, migration.destination) ||
                    (joint && *previous_owner == *candidate_owner))
                {
                    return false;
                }
                if (joint && restoration)
                {
                    const auto *original = previous->originalOwnerMap().placementFor(
                        migration.layer_idx, migration.expert_id);
                    if (!original || !matches_placement(*original, *candidate_owner))
                        return false;
                }
            }
            const auto destination = MoEExpertPlacementTarget::fromPlacement(migration.destination);
            if (!destination)
                return false;
            for (std::size_t member = 0; member < destination->participantCount(); ++member)
                ++expected_requirements[{migration.layer_idx, destination->tierIndex(),
                                         destination->participantId(member)}];
        }

        std::map<RequirementKey, size_t> actual_requirements;
        for (const auto &requirement : shadow_requirements)
        {
            if (requirement.layer_idx < 0 || requirement.tier_idx < 0 ||
                requirement.destination_participant < 0 ||
                requirement.slot_count == 0)
            {
                return false;
            }
            const RequirementKey key{
                requirement.layer_idx,
                requirement.tier_idx,
                requirement.destination_participant,
            };
            if (!actual_requirements.emplace(key, requirement.slot_count).second)
                return false;
        }
        return actual_requirements == expected_requirements;
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

    std::unique_ptr<MoEOverlayResidencyAuthority::EconomyState>
    MoEOverlayResidencyAuthority::buildEconomyState(
        const Config &config,
        const MoEOverlayResidencySnapshot &initial_snapshot)
    {
        if (config.migration_cost_profile &&
            !config.migration_economy_policy)
        {
            throw std::invalid_argument(
                "ExpertOverlay migration cost evidence requires an explicit economy policy");
        }
        if (config.migration_economy_policy)
        {
            if (config.maintenance_mode != MoERebalanceRuntimeMode::Dynamic)
            {
                throw std::invalid_argument(
                    "ExpertOverlay migration economics require Dynamic maintenance");
            }
            if (!config.migration_economy_policy->valid())
            {
                throw std::invalid_argument(
                    "ExpertOverlay migration economy policy has invalid smoothing or payoff parameters");
            }
            if (!config.phase_service_profile ||
                !config.migration_cost_profile)
            {
                throw std::invalid_argument(
                    "ExpertOverlay migration economics require complete service and movement profiles");
            }
        }

        if (config.phase_service_profile)
        {
            /*
             * Validate the setup profile with the canonical placement planner.
             * A zero-demand frozen generation checks totality, positive values,
             * and priority monotonicity without changing the incumbent plan.
             */
            DecodeExpertHistogramWindow validation_window;
            validation_window.generation = 0;
            validation_window.first_model_layer = config.model_metadata.first_model_layer;
            validation_window.num_layers = config.model_metadata.num_layers;
            validation_window.num_experts = config.model_metadata.num_experts;
            const std::size_t entries = checkedSizeProduct(
                static_cast<std::size_t>(validation_window.num_layers),
                static_cast<std::size_t>(validation_window.num_experts),
                "service-profile validation geometry");
            validation_window.expert_counts.assign(entries, 0);
            validation_window.source_expert_counts.assign(
                checkedSizeProduct(
                    entries,
                    kExpertHistogramProductionSourceCount,
                    "phase service-profile validation geometry"),
                0);
            MoERoutedExpertPlacementPlannerOptions validation_options;
            validation_options.decode_histogram_window = &validation_window;
            validation_options.phase_service_profile =
                config.phase_service_profile.get();
            validation_options.rebalancer.enabled = true;
            validation_options.rebalancer.previous_placements =
                initial_snapshot.placement_plan->placements;
            auto validation_plan = config.initial_plan;
            validation_plan.residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            (void)MoERoutedExpertPlacementPlanner::plan(
                validation_plan,
                config.model_metadata,
                validation_options);
        }

        if (!config.migration_economy_policy)
            return nullptr;

        const auto &service_profile = *config.phase_service_profile;
        const auto &migration_profile = *config.migration_cost_profile;
        if (migration_profile.identity.empty())
        {
            throw std::invalid_argument(
                "ExpertOverlay migration cost profile requires a non-empty setup identity");
        }

        auto state = std::make_unique<EconomyState>();
        state->first_model_layer = config.model_metadata.first_model_layer;
        state->num_layers = config.model_metadata.num_layers;
        state->num_experts = config.model_metadata.num_experts;
        state->tier_count = static_cast<int>(
            initial_snapshot.placement_plan->routed_tiers.size());
        if(service_profile.prepared_plan) {
            if(!service_profile.prepared_plan->authenticates(service_profile.prepared_plan))
                throw std::invalid_argument("Prepared service declaration does not retain its original metadata owner");
            for(const auto& entry:service_profile.prepared_plan->entries())state->execution_targets.push_back(entry.coordinate.target);
        }else for(const auto& participant:initial_snapshot.placementOwnerMap().participants())
            state->execution_targets.push_back(MoEOverlayRouteTarget::whole(participant.participant_id));
        std::sort(state->execution_targets.begin(),state->execution_targets.end(),[](auto a,auto b){return a.word()<b.word();});
        state->execution_targets.erase(std::unique(state->execution_targets.begin(),state->execution_targets.end(),
            [](auto a,auto b){return a.word()==b.word();}),state->execution_targets.end());
        if(state->execution_targets.empty() || state->execution_targets.size()>size_t(INT32_MAX))
            throw std::invalid_argument("Service profile has no bounded complete execution targets");
        state->participant_count=int(state->execution_targets.size());
        if(service_profile.prepared_plan) {
            const auto& original=*service_profile.prepared_plan->snapshot();
            if(original.firstModelLayer()!=initial_snapshot.firstModelLayer() || original.layerCount()!=initial_snapshot.layerCount() ||
                original.expertCount()!=initial_snapshot.expertCount() ||
                original.placementOwnerMap().participants()!=initial_snapshot.placementOwnerMap().participants() ||
                original.placementOwnerMap().owners()!=initial_snapshot.placementOwnerMap().owners())
                throw std::invalid_argument("Prepared service profile belongs to foreign original target/source geometry");
            const auto source=original.originalProjectionPreparation(),expected=initial_snapshot.originalProjectionPreparation();
            if(source.get()!=expected.get() || source.owner_before(expected) || expected.owner_before(source))
                throw std::invalid_argument("Prepared service profile has foreign original preparation ownership");
            if(config.economy_layer_catalog && (config.economy_layer_catalog.get()!=service_profile.prepared_plan->catalog().get() ||
                config.economy_layer_catalog.owner_before(service_profile.prepared_plan->catalog()) ||
                service_profile.prepared_plan->catalog().owner_before(config.economy_layer_catalog)))
                throw std::invalid_argument("Prepared service profile has a foreign source catalog");
        }else for(size_t index=0;index<state->execution_targets.size();++index)
            if(!state->execution_targets[index].isWhole() || state->execution_targets[index].word()!=int(index))
                throw std::invalid_argument("Legacy service evidence cannot certify cohort target prices");

        if (!service_profile.production_topology.valid() ||
            service_profile.production_topology.firstModelLayer() != state->first_model_layer ||
            service_profile.production_topology.layerCount() !=
                static_cast<std::size_t>(state->num_layers))
        {
            throw std::invalid_argument(
                "ExpertOverlay economy service topology disagrees with live residency geometry");
        }
        state->production_topology =
            service_profile.production_topology;
        const std::size_t service_rows = checkedSizeProduct(
            static_cast<std::size_t>(state->tier_count),
            static_cast<std::size_t>(state->num_layers),
            "service-profile tier/layer geometry");
        // Tier prices still certify placement ranking. Only participant prices
        // are retained for service scoring; there is no second live tier scorer.
        std::vector<bool> service_rows_seen(service_rows, false);
        for (const auto &row : service_profile.costs)
        {
            if (row.tier_index < 0 ||
                row.tier_index >= state->tier_count ||
                !config.model_metadata.containsModelLayer(row.layer))
            {
                throw std::invalid_argument(
                    "ExpertOverlay service profile row is outside live residency geometry");
            }
            const std::size_t offset =
                static_cast<std::size_t>(row.tier_index) *
                    static_cast<std::size_t>(state->num_layers) +
                state->storageIndexForModelLayer(row.layer);
            if (service_rows_seen[offset])
            {
                throw std::invalid_argument(
                    "ExpertOverlay service profile repeats a tier/layer row");
            }
            service_rows_seen[offset] = true;
        }
        if (service_rows_seen.size() != service_profile.costs.size() ||
            std::any_of(
                service_rows_seen.begin(),
                service_rows_seen.end(),
                [](bool seen) { return !seen; }))
        {
            throw std::invalid_argument(
                "ExpertOverlay service profile is not total for live residency geometry");
        }

        const std::size_t participant_service_rows = checkedSizeProduct(
            static_cast<std::size_t>(state->participant_count),
            static_cast<std::size_t>(state->num_layers),
            "participant service-profile geometry");
        state->participant_service_cost_rows.assign(
            participant_service_rows, nullptr);
        for (const auto &row : service_profile.target_costs)
        {
            if (row.target.isAbsent() || !config.model_metadata.containsModelLayer(row.layer))
            {
                throw std::invalid_argument(
                    "ExpertOverlay participant service profile row is outside live residency geometry");
            }
            const std::size_t offset =
                state->targetIndex(row.target) * static_cast<std::size_t>(state->num_layers) +
                state->storageIndexForModelLayer(row.layer);
            if (state->participant_service_cost_rows[offset])
            {
                throw std::invalid_argument(
                    "ExpertOverlay participant service profile repeats a participant/layer row");
            }
            for (std::size_t phase = 0;
                 phase < kExpertHistogramProductionSourceCount;
                 ++phase)
            {
                const bool has_cost =
                    row.nanoseconds_per_activation[phase] != 0u;
                if (has_cost !=
                    state->production_topology.requiresServiceEvidence(
                        row.layer, phase))
                {
                    throw std::invalid_argument(
                        "ExpertOverlay participant service profile is inconsistent with active runtime phases");
                }
            }
            state->participant_service_cost_rows[offset] = &row;
        }
        if (state->participant_service_cost_rows.size() !=
                service_profile.target_costs.size() ||
            std::any_of(
                state->participant_service_cost_rows.begin(),
                state->participant_service_cost_rows.end(),
                [](const auto *row) { return row == nullptr; }))
        {
            throw std::invalid_argument(
                "ExpertOverlay participant service profile is not total for live residency geometry");
        }

        const std::size_t participants =
            static_cast<std::size_t>(state->participant_count);
        const std::size_t participant_pairs = checkedSizeProduct(
            participants,
            participants,
            "migration-profile participant geometry");
        const std::size_t migration_slots = checkedSizeProduct(
            participant_pairs,
            static_cast<std::size_t>(state->num_layers),
            "migration-profile layer geometry");
        const std::size_t directed_pairs = checkedSizeProduct(
            participants,
            participants > 0 ? participants - 1 : 0,
            "directed migration-profile participant geometry");
        const std::size_t expected_migration_rows = checkedSizeProduct(
            directed_pairs,
            static_cast<std::size_t>(state->num_layers),
            "directed migration-profile layer geometry");
        if (migration_profile.costs.size() != expected_migration_rows)
        {
            throw std::invalid_argument(
                "ExpertOverlay migration profile must contain every directed endpoint/layer pair");
        }
        state->migration_cost_rows.assign(migration_slots, nullptr);
        for (const auto &row : migration_profile.costs)
        {
            if (!row.coordinate.valid() ||
                !config.model_metadata.containsModelLayer(row.coordinate.layer) ||
                row.transfer_and_repack_ns == 0)
            {
                throw std::invalid_argument(
                    "Migration profile has invalid original targets/layer/cost");
            }
            // Interference is a measured nonnegative cost, including positive
            // values. Retain it in the same canonical row so proposal payoff
            // charges both transfer/repack and inference interference.
            const auto source = state->targetIndex(row.coordinate.source_target);
            const auto destination = state->targetIndex(row.coordinate.destination_target);
            const auto offset =
                (source * participants + destination) *
                    static_cast<std::size_t>(state->num_layers) +
                state->storageIndexForModelLayer(row.coordinate.layer);
            if (state->migration_cost_rows[offset])
            {
                throw std::invalid_argument(
                    "Migration profile repeats an original target edge");
            }
            state->migration_cost_rows[offset] = &row;
        }
        for (int source = 0; source < state->participant_count; ++source)
        {
            for (int destination = 0;
                 destination < state->participant_count;
                 ++destination)
            {
                if (source == destination)
                    continue;
                for (int layer = 0; layer < state->num_layers; ++layer)
                {
                    const std::size_t offset =
                        (static_cast<std::size_t>(source) * participants +
                         static_cast<std::size_t>(destination)) *
                            static_cast<std::size_t>(state->num_layers) +
                        static_cast<std::size_t>(layer);
                    if (!state->migration_cost_rows[offset])
                    {
                        throw std::invalid_argument(
                            "ExpertOverlay migration profile omitted a directed endpoint/layer row");
                    }
                }
            }
        }
        state->last_moved_generation.assign(
            checkedSizeProduct(
                static_cast<std::size_t>(state->num_layers),
                static_cast<std::size_t>(state->num_experts),
                "residency-generation geometry"),
            std::numeric_limits<uint64_t>::max());
        return state;
    }

    MoEOverlayResidencyAuthority::TicketLease::TicketLease(
        MoEOverlayResidencyAuthority *authority,
        std::shared_ptr<PublishedEpochState> epoch_state,
        TicketLeasePurpose purpose)
        : authority_(authority),
          epoch_state_(std::move(epoch_state)),
          snapshot_(epoch_state_ ? epoch_state_->snapshot : nullptr),
          purpose_(purpose)
    {
    }

    MoEOverlayResidencyAuthority::TicketLease::~TicketLease()
    {
        release();
    }

    MoEOverlayResidencyAuthority::TicketLease::TicketLease(
        TicketLease &&other) noexcept
        : authority_(std::exchange(other.authority_, nullptr)),
          epoch_state_(std::move(other.epoch_state_)),
          snapshot_(std::move(other.snapshot_)),
          purpose_(other.purpose_)
    {
    }

    MoEOverlayResidencyAuthority::TicketLease &
    MoEOverlayResidencyAuthority::TicketLease::operator=(
        TicketLease &&other) noexcept
    {
        if (this == &other)
            return *this;
        release();
        authority_ = std::exchange(other.authority_, nullptr);
        epoch_state_ = std::move(other.epoch_state_);
        snapshot_ = std::move(other.snapshot_);
        purpose_ = other.purpose_;
        return *this;
    }

    void MoEOverlayResidencyAuthority::TicketLease::release() noexcept
    {
        if (authority_)
            authority_->releaseTicket(epoch_state_, purpose_);
        authority_ = nullptr;
        epoch_state_.reset();
        snapshot_.reset();
    }

    MoEOverlayResidencyAuthority::MoEOverlayResidencyAuthority(Config config)
        : config_(std::move(config)),
          planning_template_(config_.initial_plan),
          initial_histogram_window_tokens_(
              config_.histogram ? config_.histogram->windowSize() : 0)
    {
        if (config_.model_metadata.first_model_layer < 0 ||
            config_.model_metadata.num_layers <= 0 ||
            config_.model_metadata.num_experts <= 0 ||
            config_.model_metadata.num_layers > std::numeric_limits<int>::max() -
                config_.model_metadata.first_model_layer ||
            config_.initial_plan.first_model_layer != config_.model_metadata.first_model_layer)
            throw std::invalid_argument(
                "ExpertOverlay initial plan must match a valid owned model-layer interval");
        if (!config_.initial_plan.usesExpertOverlayAuthority())
        {
            throw std::invalid_argument(
                "MoE overlay residency authority requires an enabled tiered overlay");
        }
        if (jointCPUPlan(config_.initial_plan))
        {
            if (config_.maintenance_mode == MoERebalanceRuntimeMode::Off &&
                (config_.histogram || config_.participant_rebalance_policy.enabled ||
                 config_.phase_service_profile || config_.migration_cost_profile ||
                 config_.migration_economy_policy || config_.economy_layer_catalog))
                throw std::invalid_argument("Static CPU joint placement cannot own maintenance or economic state");
            if (config_.maintenance_mode == MoERebalanceRuntimeMode::Dynamic &&
                (config_.initial_plan.authority_execution != MoEOverlayAuthorityExecutionKind::HostResident ||
                 !hasTierResidencyAxis(availableMoEOptimizationMovementAxes(config_.initial_plan)) ||
                 std::none_of(config_.initial_plan.domains.begin(), config_.initial_plan.domains.end(),
                     [](const auto &domain) { return std::any_of(domain.participants.begin(), domain.participants.end(),
                         [](const auto &address) { return address.isGPU(); }); })))
                throw std::invalid_argument("Dynamic CPU cohorts require the declared mixed GPU/CPU host authority and complete original cohort targets");
        }
        if (observesHistogram() && !config_.histogram)
        {
            throw std::invalid_argument(
                "Observe or Dynamic ExpertOverlay maintenance requires a DecodeExpertHistogram");
        }
        if (config_.histogram)
        {
            if (config_.histogram->firstModelLayer() != config_.model_metadata.first_model_layer ||
                config_.histogram->config().num_layers != config_.model_metadata.num_layers ||
                config_.histogram->config().num_experts != config_.model_metadata.num_experts)
                throw std::invalid_argument("ExpertOverlay histogram must match its owned stage geometry");
            const int initial_window = config_.histogram->windowSize();
            if (!std::isfinite(
                    config_.histogram_window_growth_factor) ||
                config_.histogram_window_growth_factor <= 0.0 ||
                (config_.histogram_max_window_tokens != 0u &&
                 config_.histogram_max_window_tokens <
                     static_cast<std::uint64_t>(initial_window)) ||
                config_.histogram_max_window_tokens >
                    static_cast<std::uint64_t>(
                        std::numeric_limits<int>::max()))
            {
                throw std::invalid_argument(
                    "ExpertOverlay histogram growth requires a finite positive factor and a ceiling no smaller than its initial window");
            }
        }
        if (!config_.participant_rebalance_policy.valid())
        {
            throw std::invalid_argument(
                "ExpertOverlay participant rebalance policy has invalid thresholds or paired-entry capacity");
        }
        if (config_.migration_cost_profile &&
            !config_.migration_economy_policy)
        {
            throw std::invalid_argument(
                "ExpertOverlay migration cost evidence requires an explicit economy policy");
        }
        if (config_.migration_economy_policy)
        {
            if (!migrationEnabled())
            {
                throw std::invalid_argument(
                    "ExpertOverlay migration economics require Dynamic maintenance");
            }
            if (!config_.migration_economy_policy->valid())
            {
                throw std::invalid_argument(
                    "ExpertOverlay migration economy policy has invalid smoothing or payoff parameters");
            }
            if (!config_.phase_service_profile ||
                !config_.migration_cost_profile)
            {
                throw std::invalid_argument(
                    "ExpertOverlay migration economics require complete service and movement profiles");
            }
        }
        if (config_.economy_layer_catalog)
        {
            if (!migrationEnabled())
            {
                throw std::invalid_argument(
                    "ExpertOverlay economy layer catalog requires Dynamic maintenance");
            }
            if (config_.economy_layer_catalog->firstModelLayer() != config_.model_metadata.first_model_layer ||
                config_.economy_layer_catalog->layerCount() !=
                static_cast<std::size_t>(
                    config_.model_metadata.num_layers))
            {
                throw std::invalid_argument(
                    "ExpertOverlay economy layer catalog does not cover the complete model layer geometry");
            }
        }

        /*
         * Epoch one is the caller's exact declared layout. Successor epochs
         * use the histogram planner only when durable movement is enabled.
         * Keeping a distinct template is what permits a StaticById,
         * ExplicitMasks, or adversarial initial layout to improve over time
         * without lying about how the first physical banks were apportioned.
         */
        if (migrationEnabled())
        {
            planning_template_.residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
        }

        MoERoutedExpertPlacementPlan initial = config_.initial_plan;
        if (initial.placements.empty())
        {
            /*
             * Initial residency is deterministic by expert id even for a
             * dynamic policy. Runtime evidence cannot authorize movement before
             * any resident weights exist. A later proposal consumes the live
             * histogram and moves from this installed baseline.
             */
            MoERoutedExpertPlacementPlan initial_template = initial;
            if (isDynamicPolicy(initial_template.residency_policy))
            {
                initial_template.residency_policy =
                    RoutedExpertResidencyPolicy::StaticById;
            }
            initial = MoERoutedExpertPlacementPlanner::plan(
                          initial_template,
                          config_.model_metadata)
                          .planned_plan;
            initial.residency_policy = config_.initial_plan.residency_policy;
        }

        auto initial_snapshot = buildSnapshot(
            1,
            std::move(initial),
            config_.model_metadata,
            nullptr,nullptr,config_.projection_preparation);

        no_peer_residency_ = MoEOverlayNoPeerResidency::seal(
            *initial_snapshot->placement_plan, initial_snapshot->originalOwnerMap(), config_.model_metadata);
        economy_state_ = buildEconomyState(config_, *initial_snapshot);
        if (economy_state_ && config_.histogram)
            config_.histogram->activatePrecertifiedOptimizationDemand();
        economy_activation_state_.store(
            economy_state_
                ? EconomyActivationState::Active
                : EconomyActivationState::CollectingEvidence,
            std::memory_order_release);
        initial_snapshot_ = initial_snapshot;
        published_epoch_.store(
            std::make_shared<PublishedEpochState>(std::move(initial_snapshot)),
            std::memory_order_release);
    }

    MoEOverlayResidencyAuthority::~MoEOverlayResidencyAuthority() = default;

    std::optional<MoEOverlayResidencyAuthority::TicketLease>
    MoEOverlayResidencyAuthority::tryAcquireTicketSnapshot()
    {
        return tryAcquireCurrentSnapshot(
            TicketLeasePurpose::InferenceDispatch);
    }

    std::optional<MoEOverlayResidencyAuthority::TicketLease>
    MoEOverlayResidencyAuthority::tryAcquireGraphSequenceSnapshot()
    {
        return tryAcquireCurrentSnapshot(
            TicketLeasePurpose::InferenceGraphSequence);
    }

    MoEOverlayResidencyAuthority::TicketLease
    MoEOverlayResidencyAuthority::borrowGraphSequenceSnapshot(const TicketLease& parent)
    {
        if (parent.authority_ != this ||
            parent.purpose_ != TicketLeasePurpose::InferenceGraphSequence ||
            !parent.epoch_state_ || !parent.snapshot_)
            throw std::invalid_argument("Graph-sequence descendant requires its live original sequence parent");
        const auto state = parent.epoch_state_;
        if (state->snapshot != parent.snapshot_ ||
            published_epoch_.load(std::memory_order_acquire) != state ||
            state->active_graph_sequences.load(std::memory_order_acquire) == 0)
            throw std::logic_error("Graph-sequence parent lost its original published epoch");

        // The live parent already prevents publication. Unlike a fresh reader,
        // this descendant must not wait on the gate its own parent is draining.
        // The caller serializes parent destruction, so its nonzero ownership
        // spans all increments and closes the publisher's zero-reader race.
        state->active_tickets.fetch_add(1, std::memory_order_acq_rel);
        state->active_graph_sequences.fetch_add(1, std::memory_order_acq_rel);
        active_ticket_count_.fetch_add(1, std::memory_order_acq_rel);
        TicketLease child(this, state, TicketLeasePurpose::InferenceGraphSequence);
        PerfStatsCollector::addCounter("moe_overlay_residency", "graph_sequence_epoch_leases",
            1.0, "inference", config_.perf_device,
            {{"action", "acquire"}, {"epoch", std::to_string(child.epoch())}});
        return child;
    }
    std::optional<MoEOverlayResidencyAuthority::TicketLease>
    MoEOverlayResidencyAuthority::tryAcquireCurrentSnapshot(
        TicketLeasePurpose purpose)
    {
        for (;;)
        {
            auto candidate = published_epoch_.load(std::memory_order_acquire);
            if (!candidate || !candidate->snapshot ||
                !candidate->snapshot->valid())
            {
                throw std::logic_error(
                    "MoE overlay residency authority has no valid published snapshot");
            }

            const bool graph_sequence =
                purpose == TicketLeasePurpose::InferenceGraphSequence;
            if (graph_sequence)
            {
                const auto admission =
                    candidate->graph_sequence_admission.load(
                        std::memory_order_acquire);
                if (admission == PublishedEpochState::
                                     GraphSequenceAdmissionState::
                                         PublicationReserved)
                {
                    graph_sequence_boundary_waits_.fetch_add(
                        1, std::memory_order_relaxed);
                    PerfStatsCollector::addCounter(
                        "moe_overlay_residency",
                        "graph_sequence_boundary_waits",
                        1.0,
                        "inference",
                        config_.perf_device,
                        {{"epoch", std::to_string(candidate->snapshot->epoch)}});
                    candidate->graph_sequence_admission.wait(
                        admission, std::memory_order_acquire);
                    continue;
                }
                if (admission != PublishedEpochState::
                                     GraphSequenceAdmissionState::Open)
                {
                    ticket_acquire_retries_.fetch_add(
                        1, std::memory_order_relaxed);
                    continue;
                }
            }

            /*
             * Pin before rechecking publication. If publication won the race,
             * this provisional pin is dropped and no inference-visible lease
             * is returned. If our recheck wins, a later publisher necessarily
             * observes the pin before deciding whether the old bank can retire.
             */
            candidate->active_tickets.fetch_add(1, std::memory_order_acq_rel);
            if (graph_sequence)
            {
                candidate->active_graph_sequences.fetch_add(
                    1, std::memory_order_acq_rel);
            }
            const bool publication_unchanged =
                published_epoch_.load(std::memory_order_acquire) == candidate;
            const bool graph_sequence_still_admitted =
                !graph_sequence ||
                candidate->graph_sequence_admission.load(
                    std::memory_order_acquire) ==
                    PublishedEpochState::GraphSequenceAdmissionState::Open;
            if (publication_unchanged && graph_sequence_still_admitted)
            {
                active_ticket_count_.fetch_add(1, std::memory_order_acq_rel);
                TicketLease lease(
                    this,
                    std::move(candidate),
                    purpose);
                if (purpose == TicketLeasePurpose::InferenceGraphSequence)
                {
                    PerfStatsCollector::addCounter(
                        "moe_overlay_residency",
                        "graph_sequence_epoch_leases",
                        1.0,
                        "inference",
                        config_.perf_device,
                        {{"action", "acquire"},
                         {"epoch", std::to_string(lease.epoch())}});
                }
                return lease;
            }

            const uint64_t previous =
                candidate->active_tickets.fetch_sub(1, std::memory_order_acq_rel);
            if (previous == 0)
                std::terminate();
            if (graph_sequence)
            {
                const uint64_t previous_graph_sequences =
                    candidate->active_graph_sequences.fetch_sub(
                        1, std::memory_order_acq_rel);
                if (previous_graph_sequences == 0)
                    std::terminate();
            }
            ticket_acquire_retries_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    std::optional<MoEOverlayResidencyAuthority::TicketLease>
    MoEOverlayResidencyAuthority::tryAcquireTicketSnapshot(uint64_t epoch)
    {
        if (epoch == 0)
            return std::nullopt;

        for (;;)
        {
            auto candidate = published_epoch_.load(std::memory_order_acquire);
            if (!candidate || !candidate->snapshot ||
                candidate->snapshot->epoch != epoch)
            {
                candidate = candidate_epoch_.load(std::memory_order_acquire);
            }
            if (!candidate || !candidate->snapshot ||
                candidate->snapshot->epoch != epoch)
            {
                candidate = retiring_epoch_.load(std::memory_order_acquire);
            }
            if (!candidate || !candidate->snapshot ||
                candidate->snapshot->epoch != epoch ||
                !candidate->accepting_exact_tickets.load(
                    std::memory_order_acquire))
            {
                return std::nullopt;
            }

            /*
             * Pin before rechecking both lookup slots and the retirement gate.
             * If retirement won, this provisional count is dropped without
             * exposing a lease.  If acquisition won, retirement's final count
             * observation necessarily sees us.
             */
            candidate->active_tickets.fetch_add(1, std::memory_order_acq_rel);
            const bool still_addressable =
                candidate->accepting_exact_tickets.load(
                    std::memory_order_acquire) &&
                (published_epoch_.load(std::memory_order_acquire) == candidate ||
                 candidate_epoch_.load(std::memory_order_acquire) == candidate ||
                 retiring_epoch_.load(std::memory_order_acquire) == candidate);
            if (still_addressable)
            {
                active_ticket_count_.fetch_add(1, std::memory_order_acq_rel);
                return TicketLease(
                    this,
                    std::move(candidate),
                    TicketLeasePurpose::InferenceDispatch);
            }

            const uint64_t previous =
                candidate->active_tickets.fetch_sub(1, std::memory_order_acq_rel);
            if (previous == 0)
                std::terminate();
            ticket_acquire_retries_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    std::optional<MoEOverlayResidencyAuthority::TicketLease>
    MoEOverlayResidencyAuthority::tryAcquireCurrentBatchLLEPLease(
        int layer_idx,
        std::string_view domain_name,
        int domain_participant_count,
        std::span<uint32_t> owner_participants_out,
        std::string *error)
    {
        if (error)
            error->clear();
        const auto reject = [&](std::string message)
            -> std::optional<TicketLease>
        {
            current_batch_llep_lease_rejections_.fetch_add(
                1, std::memory_order_relaxed);
            if (error)
                *error = std::move(message);
            return std::nullopt;
        };

        if(config_.projection_preparation && config_.projection_preparation->cpuJointOwnerMap())
            return reject("Current-batch LLEP cannot reinterpret complete CPU cohort fragment ownership");
        if (layer_idx < 0 || domain_name.empty() ||
            domain_participant_count <= 1)
        {
            return reject(
                "Current-batch LLEP requires a routed layer, named domain, "
                "and at least two domain participants");
        }

        auto lease = tryAcquireTicketSnapshot();
        if (!lease || !*lease)
            return reject("Current-batch LLEP could not pin a durable epoch");
        const auto &snapshot = **lease;
        if (layer_idx >= snapshot.layered_ownership.layerCount() ||
            owner_participants_out.size() !=
                static_cast<std::size_t>(
                    snapshot.layered_ownership.expertCount()))
        {
            return reject(
                "Current-batch LLEP owner storage does not match the durable "
                "epoch geometry");
        }

        int observed_domain_participants = 0;
        for (const auto &participant : snapshot.owner_map.participants())
        {
            if (participant.domain_name != domain_name)
                continue;
            ++observed_domain_participants;
            if (participant.domain_participant_index < 0 ||
                participant.domain_participant_index >=
                    domain_participant_count)
            {
                return reject(
                    "Current-batch LLEP domain participant index is outside "
                    "the declared transaction domain");
            }
        }
        if (observed_domain_participants != domain_participant_count)
        {
            return reject(
                "Current-batch LLEP participant count disagrees with the "
                "durable epoch domain");
        }

        for (int expert = 0;
             expert < snapshot.layered_ownership.expertCount();
             ++expert)
        {
            const int global_owner =
                snapshot.layered_ownership.owner(layer_idx, expert);
            const auto *owner =
                snapshot.owner_map.participantForId(global_owner);
            if (!owner || owner->domain_name != domain_name ||
                owner->domain_participant_index < 0 ||
                owner->domain_participant_index >=
                    domain_participant_count)
            {
                return reject(
                    "Current-batch LLEP cannot borrow an expert whose durable "
                    "owner lies outside its routed domain");
            }
            owner_participants_out[static_cast<std::size_t>(expert)] =
                static_cast<uint32_t>(
                    owner->domain_participant_index);
        }

        lease->purpose_ = TicketLeasePurpose::CurrentBatchLLEP;
        current_batch_llep_leases_acquired_.fetch_add(
            1, std::memory_order_relaxed);
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "current_batch_llep_leases_acquired",
            1.0,
            "prefill",
            config_.perf_device,
            {{"epoch", std::to_string(lease->epoch())},
             {"layer", std::to_string(layer_idx)},
             {"domain", std::string(domain_name)},
             {"participants", std::to_string(domain_participant_count)}});
        return lease;
    }

    std::shared_ptr<const MoEOverlayResidencySnapshot>
    MoEOverlayResidencyAuthority::snapshot() const
    {
        const auto current = published_epoch_.load(std::memory_order_acquire);
        return current ? current->snapshot : nullptr;
    }

    RoutedExpertResidencyPolicy
    MoEOverlayResidencyAuthority::residencyPolicy() const noexcept
    {
        return config_.initial_plan.residency_policy;
    }

    MoERebalanceRuntimeMode
    MoEOverlayResidencyAuthority::maintenanceMode() const noexcept
    {
        return config_.maintenance_mode;
    }

    bool MoEOverlayResidencyAuthority::observesHistogram() const noexcept
    {
        return config_.maintenance_mode != MoERebalanceRuntimeMode::Off;
    }

    bool MoEOverlayResidencyAuthority::migrationEnabled() const noexcept
    {
        return config_.maintenance_mode == MoERebalanceRuntimeMode::Dynamic;
    }

    void MoEOverlayResidencyAuthority::
        requireHostDynamicPublicationAuthority(
            const char *operation) const
    {
        if (jointCPUPlan(config_.initial_plan) && config_.maintenance_mode == MoERebalanceRuntimeMode::Off)
            throw std::logic_error(std::string(operation) + ": Static CPU cohorts have no mutable publication authority");
        if (!migrationEnabled() ||
            config_.initial_plan.authority_execution !=
                MoEOverlayAuthorityExecutionKind::
                    DeviceResident)
        {
            return;
        }

        throw std::logic_error(
            std::string(operation ? operation : "Host ExpertOverlay publication") +
            " is forbidden because the frozen device-resident authority owns the live epoch");
    }

    void MoEOverlayResidencyAuthority::installEconomyCertification(
        std::shared_ptr<const MoERoutedTierServiceProfile> service,
        std::shared_ptr<const MoEOverlayMigrationCostProfile> migration,
        MoEOverlayMigrationEconomyPolicy policy)
    {
        requireHostDynamicPublicationAuthority(
            "Host ExpertOverlay economy certification");
        if (!migrationEnabled())
        {
            throw std::invalid_argument(
                "ExpertOverlay migration economics require Dynamic maintenance");
        }

        std::lock_guard<std::mutex> lock(economy_mutex_);
        if (checks_.load(std::memory_order_acquire) != 0)
        {
            throw std::logic_error(
                "ExpertOverlay economy certification must precede the first dynamic proposal");
        }
        if (economy_activation_state_.load(std::memory_order_acquire) !=
            EconomyActivationState::ReadyForCertification)
        {
            throw std::logic_error(
                "ExpertOverlay economy certification requires a completed calibration-demand rebase");
        }
        if (economy_state_ || config_.phase_service_profile ||
            config_.migration_cost_profile ||
            config_.migration_economy_policy)
        {
            throw std::logic_error(
                "ExpertOverlay economy certification is immutable and may be installed only once");
        }

        Config candidate_config = config_;
        candidate_config.phase_service_profile = std::move(service);
        candidate_config.migration_cost_profile = std::move(migration);
        candidate_config.migration_economy_policy = policy;
        const auto current = snapshot();
        if (!current || !current->valid())
        {
            throw std::logic_error(
                "ExpertOverlay economy certification requires a valid initial residency epoch");
        }

        /*
         * Build the complete flattened state before touching live authority
         * fields. A malformed measurement leaves the authority unchanged and
         * can therefore be diagnosed without creating a partial policy.
         */
        auto candidate_state = buildEconomyState(
            candidate_config,
            *current);
        if (!candidate_state)
        {
            throw std::invalid_argument(
                "ExpertOverlay economy certification did not contain a complete policy");
        }

        config_.phase_service_profile =
            std::move(candidate_config.phase_service_profile);
        config_.migration_cost_profile =
            std::move(candidate_config.migration_cost_profile);
        config_.migration_economy_policy = policy;
        economy_state_ = std::move(candidate_state);
        economy_activation_state_.store(
            EconomyActivationState::CertifiedAwaitingRequestBoundary,
            std::memory_order_release);

        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "economy_certifications",
            1.0,
            "model_setup",
            config_.perf_device,
            {{"service_profile", config_.phase_service_profile->identity},
             {"migration_profile", config_.migration_cost_profile->identity},
             {"immutable", "true"},
             {"blocking_inference", "false"},
             {"demand_activation", "next_request_boundary"}});
    }

    bool MoEOverlayResidencyAuthority::hasEconomyCertification() const noexcept
    {
        const auto state =
            economy_activation_state_.load(std::memory_order_acquire);
        return state ==
                   EconomyActivationState::CertifiedAwaitingRequestBoundary ||
               state == EconomyActivationState::Active;
    }

    bool MoEOverlayResidencyAuthority::optimizationDemandActive() const noexcept
    {
        return economy_activation_state_.load(std::memory_order_acquire) ==
               EconomyActivationState::Active;
    }

    MoEOverlayDemandActivationResult
    MoEOverlayResidencyAuthority::activateOptimizationDemandAtRequestBoundary()
    {
        if (!migrationEnabled())
            return MoEOverlayDemandActivationResult::NotReady;

        /*
         * Active is terminal for this authority's lifetime. The release store
         * which publishes it also publishes the opened histogram bank, so an
         * acquire read is sufficient for every subsequent request. Taking the
         * economy lock here would serialize prefill behind the entire swap
         * search on the maintenance worker, even though nothing needs changing.
         *
         * Before certification is published there is likewise no transition to
         * perform. Only the first boundary observing a complete certificate
         * takes the lock; recheck below because two callers may reach it.
         */
        const auto published =
            economy_activation_state_.load(std::memory_order_acquire);
        if (published == EconomyActivationState::Active)
            return MoEOverlayDemandActivationResult::AlreadyActive;
        if (published !=
            EconomyActivationState::CertifiedAwaitingRequestBoundary)
            return MoEOverlayDemandActivationResult::NotReady;

        std::lock_guard<std::mutex> lock(economy_mutex_);
        const auto state =
            economy_activation_state_.load(std::memory_order_acquire);
        if (state == EconomyActivationState::Active)
            return MoEOverlayDemandActivationResult::AlreadyActive;
        if (state !=
            EconomyActivationState::CertifiedAwaitingRequestBoundary)
        {
            return MoEOverlayDemandActivationResult::NotReady;
        }
        if (!economy_state_ || !config_.histogram)
        {
            throw std::logic_error(
                "Request-boundary economy activation requires an installed policy and route histogram");
        }

        config_.histogram->activateOptimizationDemand();
        economy_activation_state_.store(
            EconomyActivationState::Active,
            std::memory_order_release);
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "optimization_demand_activations",
            1.0,
            "request_admission",
            config_.perf_device,
            {{"authority", "host"},
             {"blocking", "false"},
             {"boundary", "prefill"}});
        return MoEOverlayDemandActivationResult::Activated;
    }

    bool MoEOverlayResidencyAuthority::maintenanceWindowReady() const noexcept
    {
        const auto drain_state =
            histogram_drain_state_.load(std::memory_order_acquire);
        return migrationEnabled() &&
               config_.initial_plan.authority_execution !=
                   MoEOverlayAuthorityExecutionKind::
                       DeviceResident &&
               config_.histogram &&
               (drain_state == HistogramDrainState::ProposalWindow ||
                config_.histogram->windowFull());
    }

    MoEOptimizationDemandWindow
    MoEOverlayResidencyAuthority::optimizationDemandWindow() const noexcept
    {
        return config_.histogram
                   ? config_.histogram->optimizationDemandWindow()
                   : MoEOptimizationDemandWindow{};
    }

    MoEOverlayHistogramWindowResult
    MoEOverlayResidencyAuthority::progressHistogramWindow(
        MoEOverlayHistogramEvidenceScope evidence_scope)
    {
        requireHostDynamicPublicationAuthority(
            "Host ExpertOverlay histogram progress");
        if (!migrationEnabled() || !config_.histogram)
        {
            throw std::logic_error(
                "Only Dynamic ExpertOverlay maintenance may prepare a routing window");
        }
        return progressHistogramDrain(
            HistogramDrainState::ProposalWindow, evidence_scope);
    }

    MoEOverlayHistogramRebaseProgress
    MoEOverlayResidencyAuthority::progressEconomyEvidenceRebase()
    {
        requireHostDynamicPublicationAuthority(
            "Host ExpertOverlay economy-evidence rebase");
        if (!migrationEnabled() || !config_.histogram)
        {
            throw std::logic_error(
                "Economy-evidence rebase requires one uncertified Dynamic host authority");
        }
        auto activation = economy_activation_state_.load(
            std::memory_order_acquire);
        if (activation == EconomyActivationState::ReadyForCertification)
            return MoEOverlayHistogramRebaseProgress::Complete;
        if (activation != EconomyActivationState::CollectingEvidence &&
            activation !=
                EconomyActivationState::RebasingRoutingEvidence)
        {
            throw std::logic_error(
                "Economy-evidence rebase requires one uncertified Dynamic host authority");
        }
        if (activation == EconomyActivationState::CollectingEvidence)
        {
            EconomyActivationState expected =
                EconomyActivationState::CollectingEvidence;
            if (!economy_activation_state_.compare_exchange_strong(
                    expected,
                    EconomyActivationState::RebasingRoutingEvidence,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire) &&
                expected !=
                    EconomyActivationState::RebasingRoutingEvidence)
            {
                throw std::logic_error(
                    "ExpertOverlay economy activation changed before its routing-evidence rebase");
            }
            try
            {
                config_.histogram->beginOptimizationDemandRebase();
            }
            catch (...)
            {
                economy_activation_state_.store(
                    EconomyActivationState::CollectingEvidence,
                    std::memory_order_release);
                throw;
            }
        }

        if (config_.histogram->admissionState() !=
            RuntimeExpertHistogramAdmission::CertificationQuarantine)
        {
            throw std::logic_error(
                "Economy-evidence rebase lost its route-admission quarantine");
        }

        try
        {
            const auto result = progressHistogramDrain(
                HistogramDrainState::CertificationRebase,
                MoEOverlayHistogramEvidenceScope::RuntimeSources);
            if (result.progress !=
                MoEOverlayHistogramWindowProgress::Ready)
            {
                return MoEOverlayHistogramRebaseProgress::Pending;
            }
            economy_activation_state_.store(
                EconomyActivationState::ReadyForCertification,
                std::memory_order_release);
            return MoEOverlayHistogramRebaseProgress::Complete;
        }
        catch (...)
        {
            economy_activation_state_.store(
                EconomyActivationState::CollectingEvidence,
                std::memory_order_release);
            throw;
        }
    }

    MoEOverlayHistogramWindowResult
    MoEOverlayResidencyAuthority::progressHistogramDrain(
        HistogramDrainState requested_state,
        MoEOverlayHistogramEvidenceScope evidence_scope)
    {
        if (requested_state != HistogramDrainState::ProposalWindow &&
            requested_state != HistogramDrainState::CertificationRebase)
        {
            throw std::logic_error(
                "ExpertOverlay histogram drain requires one explicit purpose");
        }

        auto current =
            histogram_drain_state_.load(std::memory_order_acquire);
        if (current == HistogramDrainState::Idle)
        {
            if (requested_state == HistogramDrainState::ProposalWindow &&
                !config_.histogram->windowFull() &&
                evidence_scope ==
                    MoEOverlayHistogramEvidenceScope::HostResident)
            {
                return {};
            }
            HistogramDrainState expected = HistogramDrainState::Idle;
            if (!histogram_drain_state_.compare_exchange_strong(
                    expected,
                    requested_state,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire))
            {
                current = expected;
            }
            else
            {
                current = requested_state;
            }
        }
        if (current != requested_state)
        {
            throw std::logic_error(
                "ExpertOverlay histogram drain purpose changed while a drain was active");
        }

        const auto drain =
            config_.histogram->progressRuntimeHistogramDrains();
        if (drain.progress ==
            RuntimeExpertHistogramDrainProgress::Pending)
        {
            return {
                .progress = MoEOverlayHistogramWindowProgress::Pending,
            };
        }
        if (drain.progress ==
            RuntimeExpertHistogramDrainProgress::Failed)
        {
            histogram_drain_state_.store(
                HistogramDrainState::Idle, std::memory_order_release);
            throw std::runtime_error(
                drain.error.empty()
                    ? "Failed to drain runtime expert histograms at the overlay maintenance boundary"
                    : drain.error);
        }
        if (!config_.histogram->syncRuntimeHistograms())
        {
            histogram_drain_state_.store(
                HistogramDrainState::Idle, std::memory_order_release);
            throw std::runtime_error(
                "Failed to merge synchronous runtime expert histograms at the overlay maintenance boundary");
        }

        /* Runtime counters can be authoritative while the host RCU bank is
         * still partial. A cadence probe reconciles and resets those device
         * banks, but it must not manufacture a proposal or advance adaptive
         * window policy until the merged host total reaches its threshold. */
        if (requested_state == HistogramDrainState::ProposalWindow &&
            !config_.histogram->windowFull())
        {
            histogram_drain_state_.store(
                HistogramDrainState::Idle, std::memory_order_release);
            return {
                .progress =
                    MoEOverlayHistogramWindowProgress::Reconciled,
            };
        }

        auto window =
            std::make_shared<DecodeExpertHistogramWindow>(
                config_.histogram->freezeAndRotateWindow());
        if (!window->valid() ||
            window->num_layers != config_.model_metadata.num_layers ||
            window->first_model_layer != config_.model_metadata.first_model_layer ||
            window->num_experts != config_.model_metadata.num_experts)
        {
            /* The completed drain has no resumable work once its bank has
             * rotated.  Return the typed lifecycle to Idle, but never advance
             * the adaptive policy from malformed evidence. */
            histogram_drain_state_.store(
                HistogramDrainState::Idle, std::memory_order_release);
            throw std::runtime_error(
                "Frozen ExpertOverlay histogram does not match model geometry");
        }
        histogram_drain_state_.store(
            HistogramDrainState::Idle, std::memory_order_release);
        return {
            .progress = MoEOverlayHistogramWindowProgress::Ready,
            .window = std::move(window),
        };
    }

    void MoEOverlayResidencyAuthority::
        advanceHistogramWindowAfterProposal(
            const MoEOverlayResidencyTransaction &transaction)
    {
        if (!config_.histogram || transaction.purpose !=
                MoEOverlayResidencyTransactionPurpose::PlacementChange ||
            !transaction.previous || !transaction.candidate ||
            !transaction.histogram_window ||
            initial_histogram_window_tokens_ <= 0)
        {
            throw std::logic_error(
                "ExpertOverlay histogram cadence requires one valid observed placement proposal");
        }

        if (transaction.empty())
        {
            const bool observed_demand =
                transaction.histogram_window->token_count != 0u ||
                std::any_of(
                    transaction.histogram_window->expert_counts.begin(),
                    transaction.histogram_window->expert_counts.end(),
                    [](const std::uint64_t count) { return count != 0u; });
            if (!observed_demand)
                return;
            growHistogramWindowAfterObservedNoMovement();
            return;
        }

        const int current = config_.histogram->windowSize();
        if (current == initial_histogram_window_tokens_)
            return;

        /* A published move changes the state evaluated by the next scan. Start
         * another short evidence window so independent tier and participant
         * objectives converge before the controller returns to cooldown. */
        config_.histogram->setWindowSize(initial_histogram_window_tokens_);
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "histogram_window_reset_after_movement",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"previous_tokens", std::to_string(current)},
             {"next_tokens",
              std::to_string(initial_histogram_window_tokens_)},
             {"histogram_generation",
              std::to_string(transaction.histogram_generation)},
             {"policy_owner", "host"},
             {"blocking_inference", "false"}});
    }

    void MoEOverlayResidencyAuthority::
        growHistogramWindowAfterObservedNoMovement()
    {
        if (!config_.histogram ||
            config_.histogram_max_window_tokens == 0u ||
            config_.histogram_window_growth_factor <= 1.0)
        {
            return;
        }

        const std::uint64_t current = static_cast<std::uint64_t>(
            config_.histogram->windowSize());
        const std::uint64_t maximum =
            config_.histogram_max_window_tokens;
        if (current >= maximum)
            return;

        const long double scaled =
            static_cast<long double>(current) *
            static_cast<long double>(
                config_.histogram_window_growth_factor);
        const std::uint64_t next =
            scaled >= static_cast<long double>(maximum)
                ? maximum
                : std::max<std::uint64_t>(
                      current + 1u,
                      static_cast<std::uint64_t>(std::ceil(scaled)));
        config_.histogram->setWindowSize(static_cast<int>(next));
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "histogram_window_growth",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"previous_tokens", std::to_string(current)},
             {"next_tokens", std::to_string(next)},
             {"maximum_tokens", std::to_string(maximum)},
             {"growth_factor",
              std::to_string(
                  config_.histogram_window_growth_factor)},
             {"policy_owner", "host"},
             {"blocking_inference", "false"}});
    }

    MoEOverlayResidencyTransaction
    MoEOverlayResidencyAuthority::proposeFromHistogram()
    {
        requireHostDynamicPublicationAuthority(
            "Host ExpertOverlay histogram proposal");
        if (!migrationEnabled())
        {
            checks_.fetch_add(1, std::memory_order_relaxed);
            const auto previous = snapshot();
            if (!previous || !previous->valid())
            {
                throw std::logic_error(
                    "Cannot propose MoE overlay residency without a valid snapshot");
            }

            MoEOverlayResidencyTransaction transaction;
            transaction.expected_epoch = previous->epoch;
            transaction.previous = previous;
            // Reuse the authority's checked unchanged-successor constructor;
            // a public epoch scalar cannot advance the private joint arm.
            transaction.candidate = buildCalibrationShadowSnapshot(previous);
            return transaction;
        }
        if (hasEconomyCertification() && !optimizationDemandActive())
        {
            throw std::logic_error(
                "ExpertOverlay cannot propose from a sealed economy profile before request-boundary demand activation");
        }

        return proposeFromFrozenHistogramWindow(
            freezeAndRotateHistogramWindow());
    }

    bool MoEOverlayResidencyAuthority::initialPreparedPlacementPublished()
        const
    {
        const auto current = snapshot();
        if (current && current->ownershipKind() == MoEOverlayResidencyOwnershipKind::CPUJointFragments)
        {
            if (!current->valid() || !initial_snapshot_ || !initial_snapshot_->valid())
                return false;
            // Epoch identity changes on restoration; the complete fixed target
            // assignments, rather than the original publication pointer, define reuse.
            const auto &live = current->placementOwnerMap().owners();
            const auto &original = initial_snapshot_->placementOwnerMap().owners();
            return live == original;
        }
        if (!current || !current->valid() || !initial_snapshot_ ||
            !initial_snapshot_->valid() ||
            current->layered_ownership !=
                initial_snapshot_->layered_ownership)
        {
            return false;
        }

        const auto &current_placements =
            current->placement_plan->placements;
        const auto &initial_placements =
            initial_snapshot_->placement_plan->placements;
        if (current_placements.size() != initial_placements.size())
            return false;
        for (std::size_t index = 0; index < current_placements.size(); ++index)
        {
            if (current_placements[index].layer !=
                    initial_placements[index].layer ||
                current_placements[index].routed_expert_tier !=
                    initial_placements[index].routed_expert_tier)
            {
                return false;
            }
        }
        return true;
    }

    MoEOverlayResidencyTransaction
    MoEOverlayResidencyAuthority::
        proposeInitialPreparedPlacementRestoration()
    {
        requireHostDynamicPublicationAuthority(
            "Host ExpertOverlay prepared-context restoration proposal");
        if (!migrationEnabled())
        {
            throw std::logic_error(
                "Prepared-context restoration requires Dynamic ExpertOverlay maintenance");
        }

        return buildInitialPreparedPlacementRestoration();
    }

    MoEOverlayResidencyTransaction
    MoEOverlayResidencyAuthority::buildInitialPreparedPlacementRestoration() const
    {
        const auto previous = snapshot();
        if (!previous || !previous->valid() || !initial_snapshot_ ||
            !initial_snapshot_->valid())
        {
            throw std::logic_error(
                "Prepared-context restoration requires valid current and initial snapshots");
        }

        const auto build_transaction =
            [&](std::shared_ptr<const MoEOverlayResidencySnapshot> candidate)
        {
            MoEOverlayResidencyTransaction result;
            result.purpose = MoEOverlayResidencyTransactionPurpose::
                PreparedContextRestoration;
            result.expected_epoch = previous->epoch;
            result.previous = previous;
            result.candidate = std::move(candidate);
            std::vector<MoELayeredExpertOwnershipChange> participant_restoration;
            if (previous->originalProjectionPreparation())
                for (const auto &destination : result.candidate->placementOwnerMap().owners())
                {
                    const auto *source = previous->placementOwnerMap().placementFor(destination.layer_idx, destination.expert_id);
                    if (source && source->tier_idx == destination.tier_idx && !source->shard_cohort &&
                        !destination.shard_cohort && source->owner_participant != destination.owner_participant)
                        participant_restoration.push_back({.layer_idx = destination.layer_idx,
                            .expert_id = destination.expert_id, .previous_participant = source->owner_participant,
                            .current_participant = destination.owner_participant});
                }
            result.migrations = buildMigrations(*previous, *result.candidate, nullptr,
                /*estimated_weight_bytes=*/0u, participant_restoration);
            result.migration_cycles = buildMoEOverlayMigrationCycles(
                result.migrations);
            result.shadow_requirements = buildMoEOverlayShadowRequirements(
                result.migrations);
            if (!result.valid())
            {
                throw std::logic_error(
                    "Prepared-context restoration produced an invalid residency transaction");
            }
            return result;
        };

        const bool joint = previous->ownershipKind() ==
                           MoEOverlayResidencyOwnershipKind::CPUJointFragments;
        auto target = joint
            ? prepareCPUJointPlacementCandidate(previous, previous->epoch,
                  initial_snapshot_->placement_plan->placements)
            : buildSnapshot(
            previous->epoch + 1u,
            *initial_snapshot_->placement_plan,
            config_.model_metadata,
            &previous->owner_map,
            &initial_snapshot_->layered_ownership);
        if (joint)
        {
            const auto changes = wholeOwnerTargetChanges(target->placementOwnerMap(),
                initial_snapshot_->placementOwnerMap().owners());
            if (!changes.empty())
                target = prepareCPUJointPlacementCandidate(previous, previous->epoch,
                    initial_snapshot_->placement_plan->placements, changes);
        }
        requireCapacityPreserving(*previous, *target);
        auto full = build_transaction(target);
        if (full.empty())
            return full;

        const auto fits_wave_budget =
            [&](const MoEOverlayResidencyTransaction &candidate)
        {
            if (config_.max_concurrent_cycles != 0u &&
                candidate.migration_cycles.size() >
                    config_.max_concurrent_cycles)
            {
                return false;
            }
            return std::all_of(
                candidate.shadow_requirements.begin(),
                candidate.shadow_requirements.end(),
                [&](const auto &requirement)
                {
                    return config_.shadow_slots_per_endpoint_layer == 0u ||
                           requirement.slot_count <=
                               config_.shadow_slots_per_endpoint_layer;
                });
        };
        if (fits_wave_budget(full))
            return full;

        /*
         * A restoration is decomposed only at closed-cycle boundaries. Start
         * from the live state and add deterministic target cycles until the
         * exact preallocated wave BOM is full. Rebuilding the candidate after
         * each addition is intentional: overlapping logical cycles may merge
         * into a different physical cycle set, and the transport must be
         * admitted against that recomposed reality.
         */
        std::vector<std::size_t> selected_cycles;
        std::optional<MoEOverlayResidencyTransaction> bounded;
        for (std::size_t cycle_index = 0;
             cycle_index < full.migration_cycles.size();
             ++cycle_index)
        {
            auto trial_cycles = selected_cycles;
            trial_cycles.push_back(cycle_index);

            MoERoutedExpertPlacementPlan trial_plan =
                *initial_snapshot_->placement_plan;
            trial_plan.placements = previous->placement_plan->placements;
            MoELayeredExpertOwnership trial_ownership = previous->layered_ownership;
            std::vector<MoEExpertOwner> exact_targets;
            for (const std::size_t selected : trial_cycles)
            {
                const auto &cycle = full.migration_cycles.at(selected);
                for (const std::size_t migration_index :
                     cycle.migration_indices)
                {
                    const auto &migration =
                        full.migrations.at(migration_index);
                    const auto placement = std::find_if(
                        trial_plan.placements.begin(),
                        trial_plan.placements.end(),
                        [&](const auto &entry)
                        { return entry.layer == migration.layer_idx; });
                    if (placement == trial_plan.placements.end() ||
                        migration.expert_id < 0 ||
                        migration.expert_id >= static_cast<int>(
                            placement->routed_expert_tier.size()))
                    {
                        throw std::logic_error(
                            "Prepared-context restoration cycle references invalid placement geometry");
                    }
                    placement->routed_expert_tier[
                        static_cast<std::size_t>(migration.expert_id)] =
                        migration.destination.tier_idx;
                    if (joint) exact_targets.push_back(migration.destination);
                    else trial_ownership.assignOwner(
                            migration.layer_idx,
                            migration.expert_id,
                            migration.destination.owner_participant);
                }
            }

            auto trial_snapshot = joint
                ? prepareCPUJointPlacementTransaction(previous, previous->epoch,
                      trial_plan.placements).candidate
                : buildSnapshot(
                previous->epoch + 1u,
                std::move(trial_plan),
                config_.model_metadata,
                &previous->owner_map,
                &trial_ownership);
            if (joint)
            {
                const auto changes = wholeOwnerTargetChanges(trial_snapshot->placementOwnerMap(), exact_targets);
                if (!changes.empty())
                    trial_snapshot = prepareCPUJointPlacementCandidate(previous, previous->epoch,
                        trial_plan.placements, changes);
            }
            requireCapacityPreserving(*previous, *trial_snapshot);
            auto trial = build_transaction(std::move(trial_snapshot));
            if (trial.empty() || !fits_wave_budget(trial))
                continue;
            selected_cycles = std::move(trial_cycles);
            bounded = std::move(trial);
            if (config_.max_concurrent_cycles != 0u &&
                bounded->migration_cycles.size() >=
                    config_.max_concurrent_cycles)
            {
                break;
            }
        }

        if (!bounded)
        {
            throw std::runtime_error(
                "Prepared-context restoration cannot express one closed cycle within the configured shadow-slot and concurrency BOM");
        }
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "prepared_context_restoration_bounded_waves",
            1.0,
            "model_teardown",
            config_.perf_device,
            {{"target_cycles", std::to_string(full.migration_cycles.size())},
             {"admitted_cycles",
              std::to_string(bounded->migration_cycles.size())},
             {"max_concurrent_cycles",
              std::to_string(config_.max_concurrent_cycles)}});
        return std::move(*bounded);
    }

    std::shared_ptr<const DecodeExpertHistogramWindow>
    MoEOverlayResidencyAuthority::freezeAndRotateHistogramWindow()
    {
        requireHostDynamicPublicationAuthority(
            "Host ExpertOverlay histogram rotation");
        if (!migrationEnabled() || !config_.histogram)
        {
            throw std::logic_error(
                "Only Dynamic ExpertOverlay maintenance may freeze a routing window");
        }
        if (!config_.histogram->syncRuntimeHistograms())
        {
            throw std::runtime_error(
                "Failed to synchronize runtime expert histograms at the overlay maintenance boundary");
        }

        auto window =
            std::make_shared<DecodeExpertHistogramWindow>(
                config_.histogram->freezeAndRotateWindow());
        if (!window->valid() ||
            window->num_layers != config_.model_metadata.num_layers ||
            window->first_model_layer != config_.model_metadata.first_model_layer ||
            window->num_experts != config_.model_metadata.num_experts)
        {
            throw std::runtime_error(
                "Frozen ExpertOverlay histogram does not match model geometry");
        }
        return window;
    }

    MoEOverlayResidencyTransaction
    MoEOverlayResidencyAuthority::proposeFromFrozenHistogramWindow(
        std::shared_ptr<const DecodeExpertHistogramWindow> window)
    {
        requireHostDynamicPublicationAuthority(
            "Host ExpertOverlay frozen-window proposal");
        if (!migrationEnabled())
        {
            throw std::logic_error(
                "ExpertOverlay histogram proposals require Dynamic maintenance");
        }
        if (hasEconomyCertification() && !optimizationDemandActive())
        {
            throw std::logic_error(
                "ExpertOverlay cannot propose from a sealed economy profile before request-boundary demand activation");
        }
        auto result = buildFromFrozenHistogramWindow(std::move(window));
        // The derivation retains the original economy lock through cadence.
        advanceHistogramWindowAfterProposal(result.transaction);
        return std::move(result.transaction);
    }

    MoEOverlayResidencyAuthority::CommittedObservation
    MoEOverlayResidencyAuthority::retainCommittedObservation(
        const MoEOverlayResidencyTransaction &committed) const
    {
        requireHostDynamicPublicationAuthority("Host ExpertOverlay committed observation");
        const auto current = snapshot();
        const CommittedObservation observation(committed.histogram_window, committed.candidate);
        std::lock_guard<std::mutex> lock(economy_mutex_);
        if (!migrationEnabled() || !committed.valid() || committed.empty() ||
            committed.purpose != MoEOverlayResidencyTransactionPurpose::PlacementChange ||
            !committed.economy.enabled || committed.economy.projected_service_gain_ns == 0u ||
            committed.economy.projected_net_benefit_ns == 0u ||
            !observation.matches(current) || !economy_state_ || !economy_state_->forecast ||
            !economy_state_->forecast->observes(committed.histogram_window))
            throw std::logic_error("ExpertOverlay continuation requires the original certified committed observation");
        return observation;
    }

    bool MoEOverlayResidencyAuthority::isCurrentCommittedObservation(
        const CommittedObservation &observation) const
    {
        return observation.matches(snapshot());
    }

    std::optional<MoEOverlayResidencyTransaction>
    MoEOverlayResidencyAuthority::reproposeFromCommittedObservation(
        const CommittedObservation &observation)
    {
        requireHostDynamicPublicationAuthority("Host ExpertOverlay observation continuation");
        if (!migrationEnabled() || !optimizationDemandActive() ||
            hasActiveBackgroundWave() || pendingRetirementCount() != 0u || pendingAbortCount() != 0u)
            throw std::logic_error("ExpertOverlay continuation requires retired original wave resources");
        auto derived = buildFromFrozenHistogramWindow(observation.window_, &observation);
        const auto &transaction = derived.transaction;
        if (transaction.empty() || !transaction.economy.enabled ||
            transaction.economy.projected_service_gain_ns == 0u ||
            transaction.economy.projected_net_benefit_ns == 0u)
        {
            // A terminal continuation is not a new demand window: in
            // particular it must not grow/reset the mutable next bank.
            recordNoMovement(false);
            return std::nullopt;
        }
        return std::move(derived.transaction);
    }

    MoEOverlayResidencyAuthority::FrozenHistogramDerivation
    MoEOverlayResidencyAuthority::buildFromFrozenHistogramWindow(
        std::shared_ptr<const DecodeExpertHistogramWindow> window,
        const CommittedObservation *continuation)
    {
        const auto current = snapshot();
        const bool joint = current && current->ownershipKind() ==
            MoEOverlayResidencyOwnershipKind::CPUJointFragments;
        // Whole-owner participant placement uses canonical typed maps even when
        // another tier is a CPU cohort; a cohort is never a scalar participant.
        // Structural candidate support still does not mint service certification.
        if(joint && config_.migration_economy_policy &&
            (!config_.phase_service_profile || !config_.phase_service_profile->prepared_plan))
            throw std::logic_error("Fixed CPU cohort candidates require original complete-target prepared service prices");
        if (!window || !window->valid() ||
            window->num_layers != config_.model_metadata.num_layers ||
            window->first_model_layer != config_.model_metadata.first_model_layer ||
            window->num_experts != config_.model_metadata.num_experts)
        {
            throw std::invalid_argument(
                "Distributed ExpertOverlay histogram window has invalid model geometry");
        }

        checks_.fetch_add(1, std::memory_order_relaxed);
        const auto previous = snapshot();
        if (!previous || !previous->valid())
        {
            throw std::logic_error(
                "Cannot propose MoE overlay residency without a valid snapshot");
        }

        MoEOverlayResidencyTransaction transaction;
        transaction.expected_epoch = previous->epoch;
        transaction.previous = previous;
        transaction.histogram_window = std::move(window);
        transaction.histogram_generation =
            transaction.histogram_window->generation;

        /*
         * Delayed setup certification uses the same lock and is permitted only
         * before the first proposal. Holding it for the complete derivation
         * makes profile publication and proposal arithmetic indivisible.
         */
        std::unique_lock<std::mutex> economy_lock(economy_mutex_);
        if (continuation && (!continuation->matches(previous) || !economy_state_ ||
            !economy_state_->forecast || !economy_state_->forecast->observes(transaction.histogram_window)))
            throw std::logic_error("ExpertOverlay continuation observation or committed epoch is stale");
        const DecodeExpertHistogramWindow *planning_window =
            transaction.histogram_window.get();
        if (config_.migration_economy_policy)
        {
            if (!economy_state_)
            {
                throw std::logic_error(
                    "ExpertOverlay migration economy state was not initialized");
            }
            auto &state = *economy_state_;
            const auto &policy = *config_.migration_economy_policy;
            if (state.forecast)
                state.forecast->observe(transaction.histogram_window, policy);
            else
                state.forecast.emplace(transaction.histogram_window);
            planning_window = &state.forecast->candidateCounts();
        }

        /*
         * Forecasts rank candidates; observations authenticate execution.
         * Keep the original window and derive migration activity from it, so
         * peers need neither a forecast copy nor a second routing payload.
         * Authenticate each borrowed view once outside the candidate loops.
         */
        const auto planning_evidence = planning_window->validatedView();
        const auto observed_evidence = transaction.histogram_window->validatedView();

        std::optional<ObservedTransactionServiceObjective> observed_service;
        if (economy_state_)
            observed_service.emplace(
                observed_evidence, economy_state_->production_topology,
                economy_state_->participant_service_cost_rows,
                economy_state_->execution_targets);

        MoERoutedExpertPlacementPlannerOptions options;
        options.decode_histogram_window = planning_window;
        options.phase_service_profile =
            config_.phase_service_profile.get();
        options.rebalancer.enabled = true;
        options.rebalancer.previous_placements =
            previous->placement_plan->placements;
        auto planned = MoERoutedExpertPlacementPlanner::plan(
            planning_template_,
            config_.model_metadata,
            options);

        // Every tier/subset candidate shares one source-bound constructor.
        // Ordinary explicit ownership remains scalar; fixed cohorts retain the
        // original preparation and complete member target through the sealed producer.
        const auto build_candidate = [&](MoERoutedExpertPlacementPlan plan,
                                         const MoELayeredExpertOwnership *ownership = nullptr,
                                         std::span<const MoEExpertOwner> exact_targets = {})
        {
            if (joint)
            {
                auto candidate = prepareCPUJointPlacementCandidate(previous, previous->epoch, plan.placements);
                const auto changes = wholeOwnerTargetChanges(candidate->placementOwnerMap(), exact_targets);
                if (!changes.empty())
                    candidate = prepareCPUJointPlacementCandidate(previous, previous->epoch,
                        std::move(plan.placements), changes);
                requireCapacityPreserving(*previous, *candidate);
                return candidate;
            }
            return buildSnapshot(previous->epoch + 1, std::move(plan),
                config_.model_metadata, &previous->owner_map, ownership);
        };
        auto tier_target_candidate = build_candidate(std::move(planned.planned_plan));

        std::vector<MoELayeredExpertOwnershipChange> participant_changes;
        const auto build_transaction =
            [&](
                std::shared_ptr<const MoEOverlayResidencySnapshot> candidate,
                std::span<const MoELayeredExpertOwnershipChange>
                    movement_objectives)
        {
            MoEOverlayResidencyTransaction result;
            result.expected_epoch = previous->epoch;
            result.histogram_generation = transaction.histogram_generation;
            result.histogram_window = transaction.histogram_window;
            result.previous = previous;
            result.candidate = std::move(candidate);
            result.migrations = buildMigrations(
                *previous,
                *result.candidate,
                &observed_evidence,
                planned.memory.routed_expert_bytes_per_expert,
                movement_objectives);
            result.migration_cycles = buildMoEOverlayMigrationCycles(
                result.migrations);
            result.shadow_requirements = buildMoEOverlayShadowRequirements(
                result.migrations);
            if (!result.valid())
            {
                throw std::logic_error(
                    "ExpertOverlay planner produced an invalid residency transaction");
            }
            return result;
        };

        const auto tier_target_transaction =
            build_transaction(tier_target_candidate, {});

        /**
         * Materialize an exact subset of closed cycles from one candidate axis.
         *
         * The subset always starts from the published epoch.  This lets a
         * one-slot scheduler compare tier and participant alternatives without
         * first composing them into a hypothetical state which can never be
         * published in that wave.  The same helper is reused by the ordinary
         * bounded composer below, keeping capacity and axis reconstruction on
         * one implementation path.
         */
        const auto build_source_cycle_subset_transaction =
            [&](const MoEOverlayResidencyTransaction &source,
                std::span<const std::size_t> cycle_indices,
                std::span<const MoELayeredExpertOwnershipChange>
                    movement_objectives)
        {
            MoERoutedExpertPlacementPlan subset_plan =
                *source.candidate->placement_plan;
            subset_plan.placements = previous->placement_plan->placements;
            MoELayeredExpertOwnership subset_ownership = previous->layered_ownership;
            std::vector<MoEExpertOwner> exact_targets;
            for (const std::size_t selected : cycle_indices)
            {
                const auto &cycle = source.migration_cycles.at(selected);
                for (const std::size_t migration_index :
                     cycle.migration_indices)
                {
                    const auto &migration =
                        source.migrations.at(migration_index);
                    const auto placement = std::find_if(
                        subset_plan.placements.begin(),
                        subset_plan.placements.end(),
                        [&](const auto &entry)
                        { return entry.layer == migration.layer_idx; });
                    if (placement == subset_plan.placements.end() ||
                        migration.expert_id < 0 ||
                        migration.expert_id >= static_cast<int>(
                            placement->routed_expert_tier.size()))
                    {
                        throw std::logic_error(
                            "ExpertOverlay bounded cycle references missing placement geometry");
                    }
                    placement->routed_expert_tier[
                        static_cast<std::size_t>(migration.expert_id)] =
                        migration.destination.tier_idx;
                    if (joint) exact_targets.push_back(migration.destination);
                    else subset_ownership.assignOwner(migration.layer_idx,
                        migration.expert_id, migration.destination.owner_participant);
                }
            }

            auto subset_snapshot = build_candidate(std::move(subset_plan), &subset_ownership, exact_targets);
            requireCapacityPreserving(*previous, *subset_snapshot);
            return build_transaction(
                std::move(subset_snapshot), movement_objectives);
        };

        /*
         * Participant makespan is a property of the epoch this bounded wave
         * can actually publish, not of the tier optimizer's eventual target.
         * Reserve one cycle for participant placement and materialize the
         * strongest publishable tier subset first. The participant planner
         * then observes precisely the owners and demand that will remain after
         * those tier cycles, so neither axis prices a hypothetical state.
         */
        std::shared_ptr<const MoEOverlayResidencySnapshot>
            participant_base_candidate = tier_target_candidate;
        std::size_t participant_base_tier_cycles =
            tier_target_transaction.migration_cycles.size();
        if (!tier_target_transaction.empty() &&
            config_.max_concurrent_cycles == 1u)
        {
            /*
             * A one-cycle wave cannot publish a tier prerequisite and a
             * participant correction together.  Derive the participant
             * alternative from the same live epoch as the tier alternative;
             * the common economy arbiter below chooses between them.  Pricing
             * participant skew against the optimizer's complete future tier
             * target would compare against a state this wave cannot reach.
             */
            participant_base_candidate = build_candidate(*previous->placement_plan,
                &previous->layered_ownership);
            participant_base_tier_cycles = 0u;
        }
        else if (!tier_target_transaction.empty() &&
            config_.max_concurrent_cycles > 1u)
        {
            struct BoundedTierCycleScore
            {
                bool eligible = true;
                uint64_t projected_service_gain_ns = 0;
                uint64_t projected_net_benefit_ns = 0;
                long double priority_weighted_demand = 0.0L;
            };

            std::vector<std::size_t> tier_cycle_order(
                tier_target_transaction.migration_cycles.size());
            std::iota(
                tier_cycle_order.begin(),
                tier_cycle_order.end(),
                0u);

            /*
             * Candidate search must use the same endpoint critical path that
             * ultimately gates admission. A raw activation winner can live on
             * a non-gating CPU participant; selecting it here and discarding
             * every other tier cycle makes the later economy gate correctly
             * reject the only candidate it is allowed to see. Score every
             * alternative once before applying the bounded-wave cut. The
             * complete transaction gate below remains the publication
             * authority and rechecks interactions with participant movement.
             */
            // One existing exact swap search per active layer, only for
            // isolated two-edge exchanges against this immutable source epoch.
            // It owns derived inputs for this proposal only; final composed
            // admission and longer cycles keep the complete-map scorer.
            int swap_layer = -1;
            std::vector<std::int32_t> swap_owners;
            std::optional<ObservedTransactionServiceObjective::SwapSearch> swap_search;
            const auto isolated_cycle_service = [&](const MoEOverlayTierMigrationCycle &cycle)
            {
                if (cycle.migration_indices.size() == 2u)
                {
                    const auto &first = tier_target_transaction.migrations.at(cycle.migration_indices[0]);
                    const auto &second = tier_target_transaction.migrations.at(cycle.migration_indices[1]);
                    if (first.layer_idx == cycle.layer_idx && second.layer_idx == cycle.layer_idx &&
                        first.expert_id != second.expert_id &&
                        economyExecutionTarget(first.source).word() == economyExecutionTarget(second.destination).word() &&
                        economyExecutionTarget(first.destination).word() == economyExecutionTarget(second.source).word())
                    {
                        if (!swap_search || swap_layer != cycle.layer_idx)
                        {
                            swap_search.reset();
                            swap_owners.assign(observed_service->numExperts(), -1);
                            for (const auto &owner : previous->placementOwnerMap().placementsForLayer(cycle.layer_idx))
                                swap_owners.at(static_cast<std::size_t>(owner.expert_id)) = observed_service->targetIndex(owner);
                            // Keep complete-map source/error validation even
                            // when all observed phases happen to be unpriced.
                            (void)observed_service->score(cycle.layer_idx, swap_owners, swap_owners);
                            swap_search = observed_service->prepareSwapSearch(cycle.layer_idx, swap_owners, {});
                            swap_layer = cycle.layer_idx;
                        }
                        // A cohort remains ONE dense certified target. Never
                        // use its absent scalar owner or split physical heat.
                        if (swap_owners.at(static_cast<std::size_t>(first.expert_id)) != observed_service->targetIndex(first.source) ||
                            swap_owners.at(static_cast<std::size_t>(second.expert_id)) != observed_service->targetIndex(second.source))
                            throw std::logic_error("ExpertOverlay economy migration source disagrees with published ownership");
                        return swap_search->scoreSwap(static_cast<std::uint32_t>(first.expert_id),
                                                      static_cast<std::uint32_t>(second.expert_id));
                    }
                }
                return scoreObservedMigrationSet(*observed_service, tier_target_transaction,
                    cycle.layer_idx, {}, cycle.migration_indices);
            };

            const auto bounded_tier_cycle_score =
                [&](const std::size_t cycle_index)
            {
                BoundedTierCycleScore score;
                const auto &cycle =
                    tier_target_transaction.migration_cycles.at(
                        cycle_index);
                for (const std::size_t migration_index :
                     cycle.migration_indices)
                {
                    const auto &migration =
                        tier_target_transaction.migrations.at(
                            migration_index);
                    const int source_priority = tierPriority(
                        *previous->placement_plan,
                        migration.source.tier_idx);
                    const int destination_priority = tierPriority(
                        *tier_target_candidate->placement_plan,
                        migration.destination.tier_idx);
                    score.priority_weighted_demand +=
                        static_cast<long double>(
                            migration.activation_count) *
                        static_cast<long double>(
                            source_priority - destination_priority);
                }

                if (!config_.migration_economy_policy)
                    return score;
                if (!economy_state_ || observed_evidence.tokenCount() == 0)
                {
                    throw std::logic_error(
                        "ExpertOverlay bounded tier selection requires complete non-zero economy evidence");
                }

                const auto &state = *economy_state_;
                const auto &policy = *config_.migration_economy_policy;
                const auto observed_cost = isolated_cycle_service(cycle);
                WideCost service_gain_one_window = 0;
                bool phase_safe = true;
                for (const auto &phase : observed_cost.by_phase)
                {
                    if (phase.after_ns > phase.before_ns)
                        phase_safe = false;
                    else
                        checkedAddWide(&service_gain_one_window,
                                       phase.before_ns - phase.after_ns,
                                       "observed bounded tier service gain");
                }

                WideCost transfer = 0;
                WideCost interference = 0;
                bool residency_eligible = true;
                const bool paired_wave_calibrated =
                    cycle.migration_indices.size() == 2u &&
                    [&]
                    {
                        const auto &first =
                            tier_target_transaction.migrations.at(
                                cycle.migration_indices[0]);
                        const auto &second =
                            tier_target_transaction.migrations.at(
                                cycle.migration_indices[1]);
                        return first.layer_idx == second.layer_idx &&
                               economyExecutionTarget(first.source).word()==economyExecutionTarget(second.destination).word() &&
                               economyExecutionTarget(first.destination).word()==economyExecutionTarget(second.source).word();
                    }();
                for (const std::size_t migration_index :
                     cycle.migration_indices)
                {
                    const auto &migration =
                        tier_target_transaction.migrations.at(
                            migration_index);
                    const auto &movement = state.migrationCost(
                        economyExecutionTarget(migration.source),
                        economyExecutionTarget(migration.destination),migration.layer_idx);
                    if (paired_wave_calibrated)
                    {
                        transfer = std::max<WideCost>(
                            transfer, movement.transfer_and_repack_ns);
                        interference = std::max<WideCost>(
                            interference,
                            movement.inference_interference_ns);
                    }
                    else
                    {
                        checkedAddWide(
                            &transfer,
                            movement.transfer_and_repack_ns,
                            "bounded tier transfer cost");
                        checkedAddWide(
                            &interference,
                            movement.inference_interference_ns,
                            "bounded tier interference cost");
                    }

                    const uint64_t last_moved =
                        state.last_moved_generation[state.expertOffset(
                            migration.layer_idx,
                            migration.expert_id)];
                    if (last_moved !=
                        std::numeric_limits<uint64_t>::max())
                    {
                        if (transaction.histogram_generation < last_moved)
                        {
                            throw std::logic_error(
                                "ExpertOverlay committed hysteresis generation exceeds bounded tier evidence");
                        }
                        residency_eligible = residency_eligible &&
                            transaction.histogram_generation - last_moved >=
                                policy.minimum_residency_generations;
                    }
                }

                const WideCost projected_gain = checkedMultiplyWide(
                    service_gain_one_window,
                    policy.payoff_horizon_tokens,
                    "bounded tier projected service gain") /
                    observed_evidence.tokenCount();
                WideCost measured_cost = transfer;
                checkedAddWide(
                    &measured_cost,
                    interference,
                    "bounded tier measured movement cost");
                const WideCost net_benefit =
                    projected_gain > measured_cost
                        ? projected_gain - measured_cost
                        : 0;
                score.projected_service_gain_ns = checkedCostToU64(
                    projected_gain,
                    "bounded tier projected service gain");
                score.projected_net_benefit_ns = checkedCostToU64(
                    net_benefit,
                    "bounded tier projected net benefit");
                score.eligible = phase_safe && residency_eligible &&
                                 net_benefit >
                                     policy.minimum_net_benefit_ns;
                return score;
            };

            std::vector<BoundedTierCycleScore> tier_cycle_scores;
            tier_cycle_scores.reserve(tier_cycle_order.size());
            for (const std::size_t cycle_index : tier_cycle_order)
            {
                tier_cycle_scores.push_back(
                    bounded_tier_cycle_score(cycle_index));
            }
            std::stable_sort(
                tier_cycle_order.begin(),
                tier_cycle_order.end(),
                [&](const std::size_t lhs, const std::size_t rhs)
                {
                    const auto &left = tier_cycle_scores.at(lhs);
                    const auto &right = tier_cycle_scores.at(rhs);
                    if (left.eligible != right.eligible)
                        return left.eligible;
                    if (left.projected_net_benefit_ns !=
                        right.projected_net_benefit_ns)
                    {
                        return left.projected_net_benefit_ns >
                               right.projected_net_benefit_ns;
                    }
                    if (left.projected_service_gain_ns !=
                        right.projected_service_gain_ns)
                    {
                        return left.projected_service_gain_ns >
                               right.projected_service_gain_ns;
                    }
                    return left.priority_weighted_demand >
                           right.priority_weighted_demand;
                });
            /*
             * Keep individually ineligible cycles behind the profitable
             * alternatives instead of deleting them. Participant makespan is
             * a max reduction: moving work off one of two co-critical
             * endpoints can have zero standalone gain yet be a necessary
             * member of a profitable bounded cohort. The complete
             * transaction economy below is the sole admission authority and
             * rejects a cohort that never becomes profitable. This ordering
             * therefore improves search without mistaking a marginal score
             * for a publication decision.
             */

            const std::size_t tier_cycle_budget =
                static_cast<std::size_t>(
                    config_.max_concurrent_cycles - 1u);
            std::vector<std::size_t> selected_tier_cycles;
            selected_tier_cycles.reserve(tier_cycle_budget);
            const auto build_tier_subset =
                [&](const std::vector<std::size_t> &cycle_indices)
            {
                MoERoutedExpertPlacementPlan subset_plan =
                    *tier_target_candidate->placement_plan;
                subset_plan.placements =
                    previous->placement_plan->placements;
                MoELayeredExpertOwnership subset_ownership = previous->layered_ownership;
                std::vector<MoEExpertOwner> exact_targets;
                for (const std::size_t cycle_index : cycle_indices)
                {
                    const auto &cycle =
                        tier_target_transaction.migration_cycles.at(
                            cycle_index);
                    for (const std::size_t migration_index :
                         cycle.migration_indices)
                    {
                        const auto &migration =
                            tier_target_transaction.migrations.at(
                                migration_index);
                        const auto placement = std::find_if(
                            subset_plan.placements.begin(),
                            subset_plan.placements.end(),
                            [&](const auto &entry)
                            { return entry.layer == migration.layer_idx; });
                        if (placement == subset_plan.placements.end() ||
                            migration.expert_id < 0 ||
                            migration.expert_id >=
                                static_cast<int>(
                                    placement->routed_expert_tier.size()))
                        {
                            throw std::logic_error(
                                "ExpertOverlay bounded tier subset references missing placement geometry");
                        }
                        placement->routed_expert_tier[
                            static_cast<std::size_t>(
                                migration.expert_id)] =
                            migration.destination.tier_idx;
                        if (joint) exact_targets.push_back(migration.destination);
                        else subset_ownership.assignOwner(migration.layer_idx,
                            migration.expert_id, migration.destination.owner_participant);
                    }
                }
                return build_candidate(std::move(subset_plan), &subset_ownership, exact_targets);
            };

            participant_base_candidate =
                build_tier_subset(selected_tier_cycles);
            std::size_t attempted_tier_cycles = 0u;
            for (const std::size_t cycle_index : tier_cycle_order)
            {
                if (selected_tier_cycles.size() >= tier_cycle_budget)
                    break;
                ++attempted_tier_cycles;
                auto trial_cycles = selected_tier_cycles;
                trial_cycles.push_back(cycle_index);
                auto trial_candidate = build_tier_subset(trial_cycles);
                const auto trial_transaction =
                    build_transaction(trial_candidate, {});
                const bool shadow_safe = std::all_of(
                    trial_transaction.shadow_requirements.begin(),
                    trial_transaction.shadow_requirements.end(),
                    [&](const auto &requirement)
                    {
                        return config_
                                       .shadow_slots_per_endpoint_layer ==
                                   0u ||
                               requirement.slot_count <=
                                   config_
                                       .shadow_slots_per_endpoint_layer;
                    });
                if (!shadow_safe)
                    continue;
                selected_tier_cycles = std::move(trial_cycles);
                participant_base_candidate =
                    std::move(trial_candidate);
            }
            participant_base_tier_cycles =
                selected_tier_cycles.size();

            // This is the original full tier axis, before participant planning
            // and final cycle recomposition. The untouched tail was not tested
            // jointly with selected work: its positive scores are ISOLATED
            // observations against `previous`, never a summed speedup/backlog.
            if (PerfStatsCollector::isRecordFamilyEnabled(
                    "moe_overlay_residency", "precut_tier_candidate_cut"))
            {
                const auto isolated_positive = [](const BoundedTierCycleScore &score)
                {
                    return score.eligible && score.projected_service_gain_ns > 0u &&
                           score.projected_net_benefit_ns > 0u;
                };
                const auto positive_count = static_cast<std::size_t>(std::count_if(
                    tier_cycle_scores.begin(), tier_cycle_scores.end(), isolated_positive));
                const auto selected_positive = static_cast<std::size_t>(std::count_if(
                    selected_tier_cycles.begin(), selected_tier_cycles.end(),
                    [&](std::size_t index) { return isolated_positive(tier_cycle_scores[index]); }));
                std::size_t cap_unvisited_positive = 0u;
                std::optional<std::size_t> best_unvisited;
                for (std::size_t position = attempted_tier_cycles;
                     position < tier_cycle_order.size(); ++position)
                {
                    const auto index = tier_cycle_order[position];
                    if (!isolated_positive(tier_cycle_scores[index]))
                        continue;
                    ++cap_unvisited_positive;
                    if (!best_unvisited)
                        best_unvisited = index; // Existing net-benefit order.
                }
                PerfStatsCollector::Tags tags{
                    {"histogram_generation", std::to_string(transaction.histogram_generation)},
                    {"source_epoch", std::to_string(previous->epoch)},
                    {"observed_window_tokens", std::to_string(observed_evidence.tokenCount())},
                    {"candidate_tier_cycles", std::to_string(tier_cycle_order.size())},
                    {"isolated_positive_tier_cycles", std::to_string(positive_count)},
                    {"selected_tier_cycles", std::to_string(selected_tier_cycles.size())},
                    {"selected_isolated_positive_tier_cycles", std::to_string(selected_positive)},
                    {"cap_unvisited_tier_cycles", std::to_string(tier_cycle_order.size() - attempted_tier_cycles)},
                    {"cap_unvisited_isolated_positive_tier_cycles", std::to_string(cap_unvisited_positive)},
                    {"shadow_rejected_before_cut", std::to_string(attempted_tier_cycles - selected_tier_cycles.size())},
                    {"tier_cycle_budget", std::to_string(tier_cycle_budget)},
                    {"maximum_concurrent_cycles", std::to_string(config_.max_concurrent_cycles)},
                    {"economy_enabled", config_.migration_economy_policy ? "true" : "false"},
                    {"score_scope", "isolated_cycle_against_observed_source_epoch"},
                    {"joint_remainder_eligibility", "not_evaluated"},
                    {"has_best_cap_unvisited_positive_cycle", best_unvisited ? "true" : "false"},
                };
                if (best_unvisited)
                {
                    const auto &score = tier_cycle_scores[*best_unvisited];
                    tags.emplace("best_cap_unvisited_cycle_index", std::to_string(*best_unvisited));
                    tags.emplace("best_cap_unvisited_isolated_projected_service_gain_ns",
                                 std::to_string(score.projected_service_gain_ns));
                    tags.emplace("best_cap_unvisited_isolated_projected_net_benefit_ns",
                                 std::to_string(score.projected_net_benefit_ns));
                }
                // The existing final projected_service_gain_ns record owns the
                // coherent admitted-transaction score. Do not sum this vector.
                PerfStatsCollector::addCounter("moe_overlay_residency",
                    "precut_tier_candidate_cut", 1.0, "maintenance",
                    config_.perf_device, std::move(tags));
            }
        }

        MoEOverlayParticipantRebalancePolicy participant_search_policy =
            config_.participant_rebalance_policy;
        std::uint32_t participant_admission_entry_limit =
            participant_search_policy.maximum_plan_entries_per_wave;
        if (participant_search_policy.enabled &&
            !tier_target_transaction.empty() &&
            config_.max_concurrent_cycles != 0u)
        {
            /*
             * Candidate breadth and physical concurrency are independent
             * contracts. Keep the caller's complete typed participant entry
             * budget here; the bounded transaction composer below is the sole
             * authority that limits accepted closed cycles. Search one
             * self-contained swap per layer: a second greedy swap in the same
             * layer depends on the first provisional owner map and is not an
             * independent alternative. Later layers still give a one-slot
             * wave enough breadth to compare both placement axes.
             */
            participant_search_policy.maximum_swaps_per_layer = 1u;
        }
        const std::optional<std::reference_wrapper<
            ObservedTransactionServiceObjective>> participant_service_objective =
            observed_service
                ? std::optional<std::reference_wrapper<
                      ObservedTransactionServiceObjective>>{
                      std::ref(*observed_service)}
                : std::nullopt;
        const auto participant_plan = planOverlayParticipantRebalance(
            *participant_base_candidate->placement_plan,
            participant_base_candidate->placementOwnerMap(),
            joint ? &previous->originalOwnerMap() : nullptr,
            planning_evidence,
            participant_search_policy,
            participant_service_objective);
        participant_rebalance_checks_.fetch_add(
            participant_plan.evidence.size(),
            std::memory_order_relaxed);
        if (participant_plan.ownerChanges() != 0u)
        {
            participant_rebalance_proposals_.fetch_add(
                1, std::memory_order_relaxed);
            participant_rebalance_owner_changes_.fetch_add(
                participant_plan.ownerChanges(),
                std::memory_order_relaxed);
        }
        for (const auto &evidence : participant_plan.evidence)
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "participant_rebalance_checks",
                1.0,
                "maintenance",
                config_.perf_device,
                {
                    {"layer", std::to_string(evidence.layer_idx)},
                    {"tier", std::to_string(evidence.tier_idx)},
                    {"service_objective_used",
                     evidence.service_objective_used ? "true" : "false"},
                    {"routed_window_activations",
                     std::to_string(
                         evidence.routed_window_activations)},
                    {"load_total", std::to_string(evidence.load_total)},
                    {"minimum_window_activations",
                     std::to_string(
                         evidence.minimum_window_activations)},
                    {"evidence_floor_satisfied",
                     evidence.routed_window_activations >=
                                 evidence.minimum_window_activations
                         ? "true"
                         : "false"},
                    {"load_min_before", std::to_string(
                                                 evidence.load_min_before)},
                    {"load_max_before", std::to_string(
                                                 evidence.load_max_before)},
                    {"load_min_after", std::to_string(
                                                evidence.load_min_after)},
                    {"load_max_after", std::to_string(
                                                evidence.load_max_after)},
                    {"swap_pairs", std::to_string(evidence.swap_pairs)},
                });
        }
        if (participant_plan.ownerChanges() != 0u)
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "participant_rebalance_owner_changes_proposed",
                static_cast<double>(participant_plan.ownerChanges()),
                "maintenance",
                config_.perf_device,
                {{"histogram_generation",
                  std::to_string(transaction.histogram_generation)}});
        }
        std::shared_ptr<const MoEOverlayResidencySnapshot> full_candidate =
            tier_target_candidate;
        std::optional<MoEOverlayResidencyTransaction>
            single_cycle_participant_transaction;
        if (participant_plan.ownerChanges() != 0u)
        {
            participant_changes = wholeOwnerTargetChanges(participant_base_candidate->placementOwnerMap(),
                participant_plan.owner_map.owners());

            /*
             * The base already contains the exact tier subset reserved for
             * this wave. Overlay the paired participant changes onto that one
             * immutable state; `buildMigrations()` may fold a newly arrived
             * expert's tier and participant destinations into one longer
             * closed cycle without ever representing an intermediate epoch.
             */
            MoELayeredExpertOwnership combined_ownership =
                participant_base_candidate->layered_ownership;
            for (const auto &change : participant_changes)
            {
                if (!joint) combined_ownership.assignOwner(
                    change.layer_idx,
                    change.expert_id,
                    change.current_participant);
            }

            auto participant_candidate = build_candidate(*participant_base_candidate->placement_plan,
                &combined_ownership, participant_plan.owner_map.owners());
            if (!tier_target_transaction.empty() &&
                config_.max_concurrent_cycles == 1u)
            {
                requireCapacityPreserving(
                    *previous, *participant_candidate);
                single_cycle_participant_transaction = build_transaction(
                    std::move(participant_candidate), participant_changes);
            }
            else
            {
                full_candidate = std::move(participant_candidate);
            }
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "tier_cycles_reserved_before_live_participant_axis",
                static_cast<double>(participant_base_tier_cycles),
                "maintenance",
                config_.perf_device,
                {{"histogram_generation",
                  std::to_string(transaction.histogram_generation)},
                 {"participant_owner_changes",
                  std::to_string(participant_changes.size())}});
        }
        requireCapacityPreserving(*previous, *full_candidate);

        const std::span<const MoELayeredExpertOwnershipChange>
            full_movement_objectives = single_cycle_participant_transaction
                                           ? std::span<const MoELayeredExpertOwnershipChange>{}
                                           : std::span<const MoELayeredExpertOwnershipChange>{participant_changes};
        auto full_transaction = build_transaction(
            full_candidate, full_movement_objectives);
        const auto fits_wave_budget = [&](
                                          const MoEOverlayResidencyTransaction &candidate)
        {
            if (config_.max_concurrent_cycles != 0 &&
                candidate.migration_cycles.size() >
                    config_.max_concurrent_cycles)
            {
                return false;
            }
            return std::all_of(
                candidate.shadow_requirements.begin(),
                candidate.shadow_requirements.end(),
                [this](const auto &requirement)
                {
                    return config_.shadow_slots_per_endpoint_layer == 0 ||
                           requirement.slot_count <=
                               config_.shadow_slots_per_endpoint_layer;
                });
        };
        /**
         * @brief Sum of observed transaction critical paths, separated by phase.
         *
         * These modeled costs use certified prices and actual route co-occurrence,
         * not a maximum of aggregate histogram work. Retain rejected costs too
         * so a phase regression cannot masquerade as a zero-benefit estimate.
         */
        struct ParticipantMakespanScore
        {
            std::array<WideCost, kExpertHistogramProductionSourceCount>
                service_before_by_phase_ns{};
            std::array<WideCost, kExpertHistogramProductionSourceCount>
                service_after_by_phase_ns{};
        };

        /** @brief The exact payoff branch, independent of residency eligibility. */
        enum class CyclePayoffDisposition
        {
            NotPriced,
            PhaseRegression,
            NoProjectedGain,
            InsufficientPayoff,
            Accepted,
        };

        /** @brief One candidate's physical cost and authoritative payoff result. */
        struct CycleEconomyScore
        {
            uint64_t projected_service_gain_ns = 0;
            uint64_t transfer_and_repack_ns = 0;
            uint64_t inference_interference_ns = 0;
            uint64_t projected_net_benefit_ns = 0;
            bool residency_eligible = true;
            CyclePayoffDisposition payoff = CyclePayoffDisposition::NotPriced;
            ParticipantMakespanScore service;

            /** @return Whether the payoff branch permits this candidate. */
            [[nodiscard]] bool payoffEligible() const noexcept
            {
                return payoff == CyclePayoffDisposition::NotPriced ||
                       payoff == CyclePayoffDisposition::Accepted;
            }

            /** @return Conjunction of independent residency and payoff policy. */
            [[nodiscard]] bool eligible() const noexcept
            {
                return residency_eligible && payoffEligible();
            }

            /** @return Stable diagnostic name, without another decision flag. */
            [[nodiscard]] const char *payoffName() const
            {
                switch (payoff)
                {
                case CyclePayoffDisposition::NotPriced: return "not_priced";
                case CyclePayoffDisposition::PhaseRegression: return "phase_regression";
                case CyclePayoffDisposition::NoProjectedGain: return "no_projected_gain";
                case CyclePayoffDisposition::InsufficientPayoff: return "insufficient_payoff";
                case CyclePayoffDisposition::Accepted: return "accepted";
                }
                throw std::logic_error("Invalid ExpertOverlay cycle payoff disposition");
            }
        };

        /** Candidate/rejection accounting retained across one-slot collapse. */
        struct SingleCycleArbitrationSummary
        {
            std::uint64_t candidate_cycles = 0u;
            std::uint64_t policy_eligible_cycles = 0u;
            MigrationCycleAxisCounts policy_eligible_axes;
            std::uint64_t individual_policy_rejected_cycles = 0u;
            std::uint64_t capacity_rejected_cycles = 0u;
            std::uint64_t capacity_rejected_migrations = 0u;
            std::uint64_t target_cycles_omitted = 0u;
            std::uint64_t target_migrations_omitted = 0u;
            std::uint64_t participant_axis_budget_rejected_cycles = 0u;
        };
        std::optional<SingleCycleArbitrationSummary>
            single_cycle_arbitration_summary;

        /** Reject phase cross-subsidy before considering aggregate payoff. */
        const auto phases_do_not_regress = [&]
            (const ParticipantMakespanScore &score)
        {
            for (std::size_t phase = 0;
                 phase < kProductionHistogramSources.size();
                 ++phase)
            {
                if (score.service_after_by_phase_ns[phase] >
                    score.service_before_by_phase_ns[phase])
                {
                    return false;
                }
            }
            return true;
        };

        /** Sum phase-pure costs only after their independent gate passes. */
        const auto aggregate_makespan = [&]
            (const std::array<
                 WideCost,
                 kExpertHistogramProductionSourceCount> &by_phase,
             const char *description)
        {
            WideCost total = 0;
            for (const WideCost phase_cost : by_phase)
            {
                checkedAddWide(
                    &total,
                    phase_cost,
                    description);
            }
            return total;
        };

        /*
         * Score every selected owner change through one participant-level
         * objective. Expert kernels for one layer run in parallel across the
         * participants and the collective cannot advance until the slowest
         * participant completes, so an individual CPU-to-GPU delta is not a
         * benefit unless it reduces that layer's actual critical path. Applying
         * tier and participant moves to the same owner map also prevents the two
         * axes from double-counting one another.
         */
        const auto score_participant_makespan = [&]
            (const MoEOverlayResidencyTransaction &candidate,
             const std::vector<std::size_t> &before_migration_indices,
             const std::vector<std::size_t> &after_migration_indices)
        {
            ParticipantMakespanScore score;
            if (after_migration_indices.empty())
                return score;

            std::vector<int> affected_layers;
            affected_layers.reserve(
                before_migration_indices.size() +
                after_migration_indices.size());
            for (const auto *indices : {
                     &before_migration_indices,
                     &after_migration_indices})
            {
                for (const std::size_t migration_index : *indices)
                {
                    const auto &migration =
                        candidate.migrations.at(migration_index);
                    affected_layers.push_back(migration.layer_idx);
                }
            }
            std::sort(affected_layers.begin(), affected_layers.end());
            affected_layers.erase(
                std::unique(affected_layers.begin(), affected_layers.end()),
                affected_layers.end());

            for (const int layer_idx : affected_layers)
            {
                const auto observed_cost = scoreObservedMigrationSet(
                    *observed_service, candidate, layer_idx,
                    before_migration_indices, after_migration_indices);
                for (std::size_t phase = 0; phase < kProductionHistogramSources.size(); ++phase)
                {
                    checkedAddWide(&score.service_before_by_phase_ns[phase],
                                   observed_cost.by_phase[phase].before_ns,
                                   "observed service before movement");
                    checkedAddWide(&score.service_after_by_phase_ns[phase],
                                   observed_cost.by_phase[phase].after_ns,
                                   "observed service after movement");
                }
            }
            return score;
        };

        /**
         * Price physical movement and hysteresis without rescoring service.
         *
         * Transaction scoring already evaluates its complete before/after
         * participant makespan once. Re-entering the service scorer for every
         * constituent cycle was both redundant and asymptotically disastrous:
         * a large model paid the complete layer/expert scan once per cycle for
         * every bounded candidate prefix.
         */
        const auto score_cycle_physical_cost = [&]
            (const MoEOverlayResidencyTransaction &candidate,
             const MoEOverlayTierMigrationCycle &cycle)
        {
            CycleEconomyScore score;
            if (!config_.migration_economy_policy)
                return score;

            const auto &policy = *config_.migration_economy_policy;
            const auto &state = *economy_state_;
            WideCost transfer = 0;
            WideCost interference = 0;
            const bool paired_wave_calibrated =
                cycle.migration_indices.size() == 2u &&
                [&]
                {
                    const auto &first = candidate.migrations.at(
                        cycle.migration_indices[0]);
                    const auto &second = candidate.migrations.at(
                        cycle.migration_indices[1]);
                    return first.layer_idx == second.layer_idx &&
                           economyExecutionTarget(first.source).word()==economyExecutionTarget(second.destination).word() &&
                           economyExecutionTarget(first.destination).word()==economyExecutionTarget(second.source).word();
                }();
            for (const std::size_t migration_index :
                 cycle.migration_indices)
            {
                const auto &migration =
                    candidate.migrations.at(migration_index);
                const auto &movement = state.migrationCost(
                    economyExecutionTarget(migration.source),
                    economyExecutionTarget(migration.destination),migration.layer_idx);
                if (paired_wave_calibrated)
                {
                    /*
                     * Pair calibration moved these reciprocal edges together.
                     * Both directed rows therefore contain the same complete
                     * gate/up/down wave critical path and interference sample;
                     * summing them would price that one wave twice.
                     */
                    transfer = std::max<WideCost>(
                        transfer, movement.transfer_and_repack_ns);
                    interference = std::max<WideCost>(
                        interference,
                        movement.inference_interference_ns);
                }
                else
                {
                    /*
                     * A longer cycle has no certified composite contention
                     * shape yet.  Add its directed pair-wave prices
                     * conservatively instead of assuming independent UPI,
                     * PCIe, network, conversion, or scheduler resources.
                     */
                    checkedAddWide(
                        &transfer,
                        movement.transfer_and_repack_ns,
                        "uncalibrated composite-cycle transfer cost");
                    checkedAddWide(
                        &interference,
                        movement.inference_interference_ns,
                        "uncalibrated composite-cycle interference");
                }

                const uint64_t last_moved =
                    state.last_moved_generation[state.expertOffset(
                        migration.layer_idx,
                        migration.expert_id)];
                if (last_moved != std::numeric_limits<uint64_t>::max())
                {
                    if (transaction.histogram_generation < last_moved)
                    {
                        throw std::logic_error(
                            "ExpertOverlay committed hysteresis generation exceeds proposal evidence");
                    }
                    if (transaction.histogram_generation - last_moved <
                        policy.minimum_residency_generations)
                    {
                        score.residency_eligible = false;
                    }
                }
            }

            score.transfer_and_repack_ns = checkedCostToU64(
                transfer,
                "projected transfer/repack cost");
            score.inference_interference_ns = checkedCostToU64(
                interference,
                "projected inference interference");
            return score;
        };

        const auto score_economy_cycle = [&]
            (const MoEOverlayResidencyTransaction &candidate,
             const MoEOverlayTierMigrationCycle &cycle)
        {
            CycleEconomyScore score =
                score_cycle_physical_cost(candidate, cycle);
            if (!config_.migration_economy_policy)
                return score;

            const auto &policy = *config_.migration_economy_policy;
            const bool advances_tier = std::any_of(
                cycle.migration_indices.begin(),
                cycle.migration_indices.end(),
                [&](const std::size_t migration_index)
                {
                    return candidate.migrations.at(migration_index)
                        .crossesTier();
                });
            std::vector<std::size_t> baseline_migration_indices;
            if (!advances_tier)
            {
                for (std::size_t migration_index = 0;
                     migration_index < candidate.migrations.size();
                     ++migration_index)
                {
                    if (candidate.migrations[migration_index].crossesTier())
                        baseline_migration_indices.push_back(migration_index);
                }
            }
            auto candidate_migration_indices = baseline_migration_indices;
            candidate_migration_indices.insert(
                candidate_migration_indices.end(),
                cycle.migration_indices.begin(),
                cycle.migration_indices.end());
            score.service = score_participant_makespan(
                candidate,
                baseline_migration_indices,
                candidate_migration_indices);
            const auto &makespan = score.service;

            if (!phases_do_not_regress(makespan))
            {
                score.payoff = CyclePayoffDisposition::PhaseRegression;
                return score;
            }
            const WideCost service_before_ns =
                aggregate_makespan(
                    makespan.service_before_by_phase_ns,
                    "aggregate participant critical path before movement");
            const WideCost service_after_ns =
                aggregate_makespan(
                    makespan.service_after_by_phase_ns,
                    "aggregate participant critical path after movement");
            const WideCost service_gain_one_window =
                service_before_ns > service_after_ns
                    ? service_before_ns - service_after_ns
                    : 0;
            if (observed_evidence.tokenCount() == 0)
            {
                throw std::logic_error(
                    "ExpertOverlay cannot score migration economy from a zero-token histogram window");
            }
            const WideCost projected_gain_numerator = checkedMultiplyWide(
                service_gain_one_window,
                policy.payoff_horizon_tokens,
                "token-horizon projected service gain");
            /*
             * Scale the observed window to a stable token lifetime. Integer
             * floor is conservative: a fractional nanosecond of projected
             * benefit can never make a migration eligible.
             */
            const WideCost projected_gain =
                projected_gain_numerator / observed_evidence.tokenCount();
            score.projected_service_gain_ns = checkedCostToU64(
                projected_gain,
                "projected service gain");
            WideCost measured_cost = score.transfer_and_repack_ns;
            checkedAddWide(
                &measured_cost,
                score.inference_interference_ns,
                "projected measured migration cost");
            if (projected_gain > measured_cost)
            {
                score.projected_net_benefit_ns = checkedCostToU64(
                    projected_gain - measured_cost,
                    "projected net benefit");
            }
            score.payoff = score.projected_net_benefit_ns > policy.minimum_net_benefit_ns
                ? CyclePayoffDisposition::Accepted
                : (projected_gain == 0 ? CyclePayoffDisposition::NoProjectedGain
                                       : CyclePayoffDisposition::InsufficientPayoff);
            return score;
        };

        /**
         * Score the selected transaction as one dependent two-axis plan.
         *
         * Promotion/demotion and within-tier changes are applied together to
         * the published owner map before measuring the participant critical
         * path. This is deliberately one objective: a faster expert endpoint
         * has no value when another participant still gates the same layer.
         * Transfer and interference prices remain conservative sums of
         * independently certified closed-cycle waves.
         */
        const auto score_economy_transaction = [&]
            (const MoEOverlayResidencyTransaction &candidate)
        {
            CycleEconomyScore score;
            if (!config_.migration_economy_policy || candidate.empty())
                return score;

            const auto &policy = *config_.migration_economy_policy;
            WideCost transfer = 0;
            WideCost interference = 0;
            std::vector<std::size_t> all_migration_indices(
                candidate.migrations.size());
            std::iota(
                all_migration_indices.begin(),
                all_migration_indices.end(),
                std::size_t{0});
            score.service = score_participant_makespan(
                candidate, {}, all_migration_indices);
            const auto &makespan = score.service;

            for (const auto &cycle : candidate.migration_cycles)
            {
                const auto cycle_score =
                    score_cycle_physical_cost(candidate, cycle);
                score.residency_eligible &=
                    cycle_score.residency_eligible;
                checkedAddWide(
                    &transfer,
                    cycle_score.transfer_and_repack_ns,
                    "transaction transfer/repack cost");
                checkedAddWide(
                    &interference,
                    cycle_score.inference_interference_ns,
                    "transaction inference interference");
            }

            if (observed_evidence.tokenCount() == 0)
            {
                throw std::logic_error(
                    "ExpertOverlay cannot score transaction economy from a zero-token histogram window");
            }
            if (!phases_do_not_regress(makespan))
            {
                score.payoff = CyclePayoffDisposition::PhaseRegression;
                return score;
            }
            const WideCost service_before_ns =
                aggregate_makespan(
                    makespan.service_before_by_phase_ns,
                    "aggregate participant critical path before movement");
            const WideCost service_after_ns =
                aggregate_makespan(
                    makespan.service_after_by_phase_ns,
                    "aggregate participant critical path after movement");
            const WideCost service_gain_one_window =
                service_before_ns > service_after_ns
                    ? service_before_ns - service_after_ns
                    : 0;
            const WideCost projected_service_gain =
                checkedMultiplyWide(
                    service_gain_one_window,
                    policy.payoff_horizon_tokens,
                    "transaction token-horizon service gain") /
                observed_evidence.tokenCount();
            WideCost measured_cost = transfer;
            checkedAddWide(
                &measured_cost,
                interference,
                "transaction measured migration cost");

            score.projected_service_gain_ns = checkedCostToU64(
                projected_service_gain,
                "transaction service gain");
            score.transfer_and_repack_ns = checkedCostToU64(
                transfer,
                "transaction transfer/repack cost");
            score.inference_interference_ns = checkedCostToU64(
                interference,
                "transaction inference interference");
            if (projected_service_gain > measured_cost)
            {
                score.projected_net_benefit_ns = checkedCostToU64(
                    projected_service_gain - measured_cost,
                    "transaction net benefit");
            }
            score.payoff = score.projected_net_benefit_ns > policy.minimum_net_benefit_ns
                ? CyclePayoffDisposition::Accepted
                : (projected_service_gain == 0 ? CyclePayoffDisposition::NoProjectedGain
                                               : CyclePayoffDisposition::InsufficientPayoff);
            return score;
        };

        if (single_cycle_participant_transaction)
        {
            /** One independently publishable alternative for the sole lane. */
            struct SingleCycleAlternative
            {
                const MoEOverlayResidencyTransaction *source = nullptr;
                std::size_t cycle_index = 0u;
                MoEOptimizationMovementAxis axis =
                    MoEOptimizationMovementAxis::TierResidency;
                CycleEconomyScore economy;
                long double priority_weighted_demand = 0.0L;
                std::uint32_t participant_objective_entries = 0u;
                bool economy_eligible = false;
                bool participant_budget_safe = false;

                /** @return Whether every policy gate admits this alternative. */
                [[nodiscard]] bool admissionEligible() const noexcept
                {
                    return economy_eligible && participant_budget_safe;
                }
            };

            std::vector<SingleCycleAlternative> alternatives;
            alternatives.reserve(
                tier_target_transaction.migration_cycles.size() +
                single_cycle_participant_transaction->migration_cycles.size());
            const auto append_alternatives = [&]
                (const MoEOverlayResidencyTransaction &source)
            {
                for (std::size_t cycle_index = 0u;
                     cycle_index < source.migration_cycles.size();
                     ++cycle_index)
                {
                    const auto &cycle =
                        source.migration_cycles[cycle_index];
                    SingleCycleAlternative alternative{
                        .source = &source,
                        .cycle_index = cycle_index,
                        .axis = migrationCycleAxis(source, cycle),
                    };
                    for (const std::size_t migration_index :
                         cycle.migration_indices)
                    {
                        const auto &migration =
                            source.migrations.at(migration_index);
                        const int source_priority = tierPriority(
                            *previous->placement_plan,
                            migration.source.tier_idx);
                        const int destination_priority = tierPriority(
                            *source.candidate->placement_plan,
                            migration.destination.tier_idx);
                        alternative.priority_weighted_demand +=
                            static_cast<long double>(
                                migration.activation_count) *
                            static_cast<long double>(
                                source_priority - destination_priority);
                        if (advancesParticipantPlacement(migration.axis))
                        {
                            ++alternative.participant_objective_entries;
                        }
                    }
                    alternative.economy = score_economy_cycle(
                        source, cycle);
                    alternative.participant_budget_safe =
                        alternative.participant_objective_entries <=
                        participant_admission_entry_limit;
                    alternative.economy_eligible =
                        !config_.migration_economy_policy ||
                        alternative.economy.eligible();
                    alternatives.push_back(std::move(alternative));
                }
            };
            append_alternatives(tier_target_transaction);
            append_alternatives(*single_cycle_participant_transaction);

            std::stable_sort(
                alternatives.begin(),
                alternatives.end(),
                [&](const auto &lhs, const auto &rhs)
                {
                    if (lhs.admissionEligible() != rhs.admissionEligible())
                        return lhs.admissionEligible();
                    if (config_.migration_economy_policy)
                    {
                        if (lhs.economy.projected_net_benefit_ns !=
                            rhs.economy.projected_net_benefit_ns)
                        {
                            return lhs.economy.projected_net_benefit_ns >
                                   rhs.economy.projected_net_benefit_ns;
                        }
                        if (lhs.economy.projected_service_gain_ns !=
                            rhs.economy.projected_service_gain_ns)
                        {
                            return lhs.economy.projected_service_gain_ns >
                                   rhs.economy.projected_service_gain_ns;
                        }
                    }
                    else if (lhs.priority_weighted_demand !=
                             rhs.priority_weighted_demand)
                    {
                        return lhs.priority_weighted_demand >
                               rhs.priority_weighted_demand;
                    }
                    return lhs.cycle_index < rhs.cycle_index;
                });

            std::optional<SingleCycleAlternative> selected_alternative;
            std::optional<MoEOverlayResidencyTransaction>
                selected_transaction;
            for (const bool require_policy_eligible : {true, false})
            {
                for (const auto &alternative : alternatives)
                {
                    if (alternative.admissionEligible() !=
                        require_policy_eligible)
                    {
                        continue;
                    }
                    const std::array<std::size_t, 1> cycle_indices{
                        alternative.cycle_index};
                    const bool participant_alternative =
                        alternative.source ==
                        &*single_cycle_participant_transaction;
                    const std::span<
                        const MoELayeredExpertOwnershipChange>
                        objectives = participant_alternative
                                         ? std::span<const MoELayeredExpertOwnershipChange>{participant_changes}
                                         : std::span<const MoELayeredExpertOwnershipChange>{};
                    auto exact = build_source_cycle_subset_transaction(
                        *alternative.source, cycle_indices, objectives);
                    if (!fits_wave_budget(exact))
                        continue;
                    selected_alternative = alternative;
                    selected_transaction = std::move(exact);
                    break;
                }
                if (selected_transaction)
                    break;
            }
            if (!selected_transaction || !selected_alternative)
            {
                throw std::runtime_error(
                    "ExpertOverlay one-cycle arbitration found no candidate compatible with the physical shadow-slot BOM");
            }

            std::size_t tier_candidates = 0u;
            std::size_t participant_candidates = 0u;
            std::size_t eligible_tier_candidates = 0u;
            std::size_t eligible_participant_candidates = 0u;
            SingleCycleArbitrationSummary arbitration_summary;
            arbitration_summary.candidate_cycles = alternatives.size();
            for (const auto &alternative : alternatives)
            {
                const bool advances_participant =
                    advancesParticipantPlacement(alternative.axis);
                if (advances_participant)
                {
                    ++participant_candidates;
                    if (alternative.admissionEligible())
                        ++eligible_participant_candidates;
                }
                if (advancesTierResidency(alternative.axis))
                {
                    ++tier_candidates;
                    if (alternative.admissionEligible())
                        ++eligible_tier_candidates;
                }
                if (!alternative.economy_eligible)
                {
                    ++arbitration_summary
                          .individual_policy_rejected_cycles;
                    continue;
                }
                ++arbitration_summary.policy_eligible_cycles;
                arbitration_summary.policy_eligible_axes.add(
                    alternative.axis);
                if (!alternative.participant_budget_safe)
                {
                    ++arbitration_summary
                          .participant_axis_budget_rejected_cycles;
                    continue;
                }
                const bool selected =
                    alternative.source == selected_alternative->source &&
                    alternative.cycle_index ==
                        selected_alternative->cycle_index;
                if (!selected)
                {
                    ++arbitration_summary.capacity_rejected_cycles;
                    arbitration_summary.capacity_rejected_migrations +=
                        alternative.source->migration_cycles
                            .at(alternative.cycle_index)
                            .migration_indices.size();
                    /*
                     * The process-level `target_*_omitted` counters predate
                     * independent-axis arbitration and describe how much of
                     * the tier-residency target was deferred by the physical
                     * wave BOM. A participant-only contender is a competing
                     * policy alternative, not another edge in that target.
                     * Keep it in the typed admission proof above, but do not
                     * double-count it as omitted tier work.
                     */
                    if (alternative.source == &tier_target_transaction)
                    {
                        ++arbitration_summary.target_cycles_omitted;
                        arbitration_summary.target_migrations_omitted +=
                            alternative.source->migration_cycles
                                .at(alternative.cycle_index)
                                .migration_indices.size();
                    }
                }
            }
            single_cycle_arbitration_summary = arbitration_summary;
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "single_cycle_axis_arbitration",
                1.0,
                "maintenance",
                config_.perf_device,
                {
                    {"tier_candidates", std::to_string(tier_candidates)},
                    {"participant_candidates",
                     std::to_string(participant_candidates)},
                    {"eligible_tier_candidates",
                     std::to_string(eligible_tier_candidates)},
                    {"eligible_participant_candidates",
                     std::to_string(eligible_participant_candidates)},
                    {"selected_axis",
                     movementAxisName(selected_alternative->axis)},
                    {"selected_policy_eligible",
                     selected_alternative->admissionEligible()
                         ? "true"
                         : "false"},
                    {"selected_projected_net_benefit_ns",
                     std::to_string(
                         selected_alternative->economy
                             .projected_net_benefit_ns)},
                    {"selection_policy",
                     config_.migration_economy_policy
                         ? "measured_economy"
                         : "priority_weighted_demand"},
                });

            const bool selected_participant =
                advancesParticipantPlacement(
                    selected_alternative->axis);
            if (!selected_participant)
                participant_changes.clear();
            full_candidate = selected_transaction->candidate;
            full_transaction = std::move(*selected_transaction);
        }

        MoEOverlayMigrationEconomyEvidence economy_evidence;
        std::vector<CycleEconomyScore> full_cycle_economy;
        struct RejectedPayoffEnvelope
        {
            std::size_t cycle_index = 0;
            std::size_t migration_count = 0;
            CycleEconomyScore score;
            WideCost shortfall_ns = 0;
        };
        std::optional<RejectedPayoffEnvelope> closest_rejected_payoff;
        if (config_.migration_economy_policy)
        {
            const auto &policy = *config_.migration_economy_policy;
            economy_evidence.enabled = true;
            economy_evidence.service_profile_identity =
                config_.phase_service_profile->identity;
            economy_evidence.migration_profile_identity =
                config_.migration_cost_profile->identity;
            economy_evidence.smoothed_through_generation =
                transaction.histogram_generation;
            economy_evidence.forecast_fingerprint = economy_state_->forecast->fingerprint();
            economy_evidence.historical_window_weight =
                policy.historical_window_weight;
            economy_evidence.current_window_weight =
                policy.current_window_weight;
            economy_evidence.payoff_horizon_tokens =
                policy.payoff_horizon_tokens;
            economy_evidence.minimum_net_benefit_ns =
                policy.minimum_net_benefit_ns;
            economy_evidence.minimum_residency_generations =
                policy.minimum_residency_generations;
            full_cycle_economy.reserve(
                full_transaction.migration_cycles.size());
            for (std::size_t cycle_index = 0;
                 cycle_index < full_transaction.migration_cycles.size();
                 ++cycle_index)
            {
                const auto &cycle =
                    full_transaction.migration_cycles[cycle_index];
                auto score = score_economy_cycle(
                    full_transaction, cycle);
                economy_evidence.residency_rejected_cycles +=
                    score.residency_eligible ? 0u : 1u;
                economy_evidence.payoff_rejected_cycles +=
                    score.payoffEligible() ? 0u : 1u;
                // Mirror the already computed decision. Observability cannot
                // select a candidate or cause any device/route materialization.
                if (PerfStatsCollector::isDomainEnabled("moe_overlay_residency"))
                {
                    for (std::size_t phase = 0; phase < kProductionHistogramSources.size(); ++phase)
                    {
                        const PerfStatsCollector::Tags cycle_tags{
                            {"histogram_generation", std::to_string(full_transaction.histogram_generation)},
                            {"cycle_index", std::to_string(cycle_index)},
                            {"source", phase == 0 ? "decode" : (phase == 1 ? "prefill" : "grouped_verifier")},
                            {"payoff_disposition", score.payoffName()},
                            {"migration_count", std::to_string(cycle.migration_indices.size())},
                            {"service_profile", config_.phase_service_profile->identity},
                            {"migration_profile", config_.migration_cost_profile->identity},
                        };
                        PerfStatsCollector::addCounter("moe_overlay_residency", "economy_cycle_service_before_ns",
                            static_cast<double>(score.service.service_before_by_phase_ns[phase]),
                            "maintenance", config_.perf_device, cycle_tags);
                        PerfStatsCollector::addCounter("moe_overlay_residency", "economy_cycle_service_after_ns",
                            static_cast<double>(score.service.service_after_by_phase_ns[phase]),
                            "maintenance", config_.perf_device, cycle_tags);
                    }
                }
                if (!score.payoffEligible())
                {
                    /*
                     * Preserve one internally consistent rejected cycle, not
                     * independent maxima from unrelated cycles.  Operators can
                     * therefore distinguish an inadequate payoff horizon from
                     * transfer or interference regressions using PerfStats
                     * without enabling verbose per-cycle logging.
                     */
                    WideCost required_gain = score.transfer_and_repack_ns;
                    checkedAddWide(
                        &required_gain,
                        score.inference_interference_ns,
                        "rejected-cycle measured cost");
                    checkedAddWide(
                        &required_gain,
                        policy.minimum_net_benefit_ns,
                        "rejected-cycle minimum net benefit");
                    checkedAddWide(
                        &required_gain,
                        1,
                        "rejected-cycle strict payoff threshold");
                    const WideCost shortfall =
                        required_gain > score.projected_service_gain_ns
                            ? required_gain -
                                  score.projected_service_gain_ns
                            : 0;
                    if (!closest_rejected_payoff ||
                        shortfall < closest_rejected_payoff->shortfall_ns ||
                        (shortfall ==
                             closest_rejected_payoff->shortfall_ns &&
                         score.projected_service_gain_ns >
                             closest_rejected_payoff->score
                                 .projected_service_gain_ns))
                    {
                        closest_rejected_payoff = RejectedPayoffEnvelope{
                            .cycle_index = cycle_index,
                            .migration_count =
                                cycle.migration_indices.size(),
                            .score = score,
                            .shortfall_ns = shortfall,
                        };
                    }
                }
                full_cycle_economy.push_back(score);
            }
        }

        const auto finalize_economy = [&]
            (MoEOverlayResidencyTransaction candidate)
        {
            if (!config_.migration_economy_policy)
            {
                return FrozenHistogramDerivation{std::move(candidate), std::move(economy_lock)};
            }

            auto evidence = economy_evidence;
            const auto transaction_score =
                score_economy_transaction(candidate);
            if (!candidate.empty() && !transaction_score.eligible())
            {
                throw std::logic_error(
                    "ExpertOverlay selected a dependent migration transaction rejected by its economy policy");
            }
            evidence.projected_service_gain_ns =
                transaction_score.projected_service_gain_ns;
            evidence.projected_transfer_and_repack_ns =
                transaction_score.transfer_and_repack_ns;
            evidence.projected_inference_interference_ns =
                transaction_score.inference_interference_ns;
            evidence.projected_net_benefit_ns =
                transaction_score.projected_net_benefit_ns;
            candidate.economy = std::move(evidence);
            if (!candidate.valid())
            {
                throw std::logic_error(
                    "ExpertOverlay economy policy produced invalid transaction evidence");
            }

            economy_proposals_.fetch_add(1, std::memory_order_relaxed);
            payoff_rejected_cycles_.fetch_add(
                candidate.economy.payoff_rejected_cycles,
                std::memory_order_relaxed);
            residency_rejected_cycles_.fetch_add(
                candidate.economy.residency_rejected_cycles,
                std::memory_order_relaxed);
            const PerfStatsCollector::Tags tags{
                {"service_profile",
                 candidate.economy.service_profile_identity},
                {"migration_profile",
                 candidate.economy.migration_profile_identity},
                {"histogram_generation",
                 std::to_string(candidate.histogram_generation)},
                {"observed_window_tokens",
                 std::to_string(candidate.histogram_window->token_count)},
                {"forecast_window_tokens",
                 std::to_string(planning_window->token_count)},
                {"forecast_fingerprint",
                 std::to_string(candidate.economy.forecast_fingerprint)},
            };
            const auto add = [&](const char *name, uint64_t value)
            {
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    name,
                    static_cast<double>(value),
                    "maintenance",
                    config_.perf_device,
                    tags);
            };
            add("economy_proposals", 1);
            add("projected_service_gain_ns",
                candidate.economy.projected_service_gain_ns);
            add("projected_transfer_and_repack_ns",
                candidate.economy.projected_transfer_and_repack_ns);
            add("projected_inference_interference_ns",
                candidate.economy.projected_inference_interference_ns);
            add("projected_net_benefit_ns",
                candidate.economy.projected_net_benefit_ns);
            add("payoff_rejected_cycles",
                candidate.economy.payoff_rejected_cycles);
            add("residency_rejected_cycles",
                candidate.economy.residency_rejected_cycles);
            if (closest_rejected_payoff)
            {
                auto rejected_tags = tags;
                rejected_tags.emplace("payoff_disposition", closest_rejected_payoff->score.payoffName());
                rejected_tags.emplace(
                    "cycle_index",
                    std::to_string(
                        closest_rejected_payoff->cycle_index));
                rejected_tags.emplace(
                    "migration_count",
                    std::to_string(
                        closest_rejected_payoff->migration_count));
                rejected_tags.emplace(
                    "histogram_window_tokens",
                    std::to_string(candidate.histogram_window->token_count));
                rejected_tags.emplace(
                    "payoff_horizon_tokens",
                    std::to_string(
                        candidate.economy.payoff_horizon_tokens));
                const auto add_rejected =
                    [&](const char *name, long double value)
                {
                    PerfStatsCollector::addCounter(
                        "moe_overlay_residency",
                        name,
                        static_cast<double>(value),
                        "maintenance",
                        config_.perf_device,
                        rejected_tags);
                };
                add_rejected(
                    "closest_rejected_projected_service_gain_ns",
                    closest_rejected_payoff->score
                        .projected_service_gain_ns);
                add_rejected(
                    "closest_rejected_transfer_and_repack_ns",
                    closest_rejected_payoff->score
                        .transfer_and_repack_ns);
                add_rejected(
                    "closest_rejected_inference_interference_ns",
                    closest_rejected_payoff->score
                        .inference_interference_ns);
                add_rejected(
                    "closest_rejected_payoff_shortfall_ns",
                    static_cast<long double>(
                        closest_rejected_payoff->shortfall_ns));
            }
            return FrozenHistogramDerivation{std::move(candidate), std::move(economy_lock)};
        };

        if (full_transaction.empty())
            return finalize_economy(std::move(full_transaction));

        /*
         * A physical cycle may carry a tier exchange, a participant objective,
         * or both. Count only the participant planner's explicit owner deltas
         * when enforcing its policy budget; a cross-tier edge must not consume
         * a fictional same-tier entry merely because both objectives later
         * share one closed transfer cycle.
         */
        const auto participant_objective_entries = [&]
            (const std::size_t cycle_index)
        {
            std::uint32_t entries = 0u;
            const auto &cycle =
                full_transaction.migration_cycles.at(cycle_index);
            for (const std::size_t migration_index :
                 cycle.migration_indices)
            {
                const auto &migration =
                    full_transaction.migrations.at(migration_index);
                const bool is_participant_objective = std::any_of(
                    participant_changes.begin(),
                    participant_changes.end(),
                    [&](const auto &change)
                    {
                        return change.layer_idx == migration.layer_idx &&
                               change.expert_id == migration.expert_id &&
                               change.current_participant ==
                                   migration.destination.owner_participant;
                    });
                if (is_participant_objective)
                    ++entries;
            }
            return entries;
        };

        /**
         * @brief Materialize one exact subset of candidate closed cycles.
         *
         * The subset starts from the published epoch and applies both typed
         * placement axes together. Rebuilding through the canonical snapshot
         * constructor is intentional: it proves capacity conservation and
         * exposes any physical cycle recomposition before economy admission.
         */
        const auto build_cycle_subset_transaction = [&]
            (const std::vector<std::size_t> &cycle_indices)
        {
            return build_source_cycle_subset_transaction(
                full_transaction,
                cycle_indices,
                participant_changes);
        };

        /*
         * Without measured economics, retain the historical priority-weighted
         * bounded-wave order. With economics, rank only eligible cycles by
         * exact projected net benefit; expert id remains the stable final tie.
         */
        std::vector<std::size_t> cycle_order(
            0);
        cycle_order.reserve(full_transaction.migration_cycles.size());
        for (std::size_t cycle_index = 0;
             cycle_index < full_transaction.migration_cycles.size();
             ++cycle_index)
        {
            if (!config_.migration_economy_policy ||
                full_cycle_economy[cycle_index].eligible())
            {
                cycle_order.push_back(cycle_index);
            }
        }
        const auto cycle_score = [&](std::size_t cycle_index)
        {
            long double score = 0.0L;
            const auto &cycle =
                full_transaction.migration_cycles[cycle_index];
            for (const std::size_t migration_index : cycle.migration_indices)
            {
                const auto &migration =
                    full_transaction.migrations[migration_index];
                const int source_priority = tierPriority(
                    *previous->placement_plan,
                    migration.source.tier_idx);
                const int destination_priority = tierPriority(
                    *full_candidate->placement_plan,
                    migration.destination.tier_idx);
                score += static_cast<long double>(migration.activation_count) *
                         static_cast<long double>(
                             source_priority - destination_priority);
            }
            return score;
        };
        std::stable_sort(
            cycle_order.begin(),
            cycle_order.end(),
            [&](std::size_t lhs, std::size_t rhs)
            {
                if (config_.migration_economy_policy)
                {
                    const auto &left = full_cycle_economy[lhs];
                    const auto &right = full_cycle_economy[rhs];
                    if (left.projected_net_benefit_ns !=
                        right.projected_net_benefit_ns)
                    {
                        return left.projected_net_benefit_ns >
                               right.projected_net_benefit_ns;
                    }
                    if (left.projected_service_gain_ns !=
                        right.projected_service_gain_ns)
                    {
                        return left.projected_service_gain_ns >
                               right.projected_service_gain_ns;
                    }
                    return lhs < rhs;
                }
                return cycle_score(lhs) > cycle_score(rhs);
            });

        /**
         * @brief An atomically admitted set whose members do not pay alone.
         *
         * Critical-path economics is not separable. Two co-critical endpoints
         * can require a tier exchange and a participant exchange before the
         * maximum decreases at all. Participant-cycle scoring deliberately
         * uses the reserved tier state as its baseline, so admission must
         * retain that dependency instead of filtering the tier prerequisite by
         * its zero standalone payoff.
         */
        struct DependentCycleCohort
        {
            std::vector<std::size_t> cycle_indices;
            MoEOverlayResidencyTransaction transaction;
            CycleEconomyScore economy;
            std::uint32_t participant_entries = 0u;
        };
        std::optional<DependentCycleCohort> dependent_cycle_cohort;
        std::uint64_t dependent_cohort_candidates = 0u;
        std::uint64_t dependent_cohort_snapshot_builds = 0u;
        std::uint64_t dependent_cohort_payoff_rejections = 0u;
        if (config_.migration_economy_policy)
        {
            const std::size_t cycle_capacity =
                config_.max_concurrent_cycles == 0u
                    ? full_transaction.migration_cycles.size()
                    : static_cast<std::size_t>(
                          config_.max_concurrent_cycles);
            std::vector<std::size_t> tier_prerequisites;
            std::vector<std::size_t> participant_candidates;
            for (std::size_t cycle_index = 0u;
                 cycle_index < full_transaction.migration_cycles.size();
                 ++cycle_index)
            {
                const auto axis = migrationCycleAxis(
                    full_transaction,
                    full_transaction.migration_cycles[cycle_index]);
                if (advancesTierResidency(axis))
                    tier_prerequisites.push_back(cycle_index);
                if (!advancesTierResidency(axis) &&
                    advancesParticipantPlacement(axis) &&
                    full_cycle_economy[cycle_index].eligible())
                {
                    participant_candidates.push_back(cycle_index);
                }
            }

            std::set<std::vector<std::size_t>> candidate_cohorts;
            for (const std::size_t participant : participant_candidates)
            {
                const int dependency_layer =
                    full_transaction.migration_cycles[participant]
                        .layer_idx;
                std::vector<std::size_t> candidate;
                bool has_rejected_tier_prerequisite = false;
                for (const std::size_t tier_cycle : tier_prerequisites)
                {
                    if (full_transaction
                            .migration_cycles[tier_cycle]
                            .layer_idx != dependency_layer)
                    {
                        continue;
                    }
                    candidate.push_back(tier_cycle);
                    has_rejected_tier_prerequisite =
                        has_rejected_tier_prerequisite ||
                        !full_cycle_economy[tier_cycle].eligible();
                }
                if (!has_rejected_tier_prerequisite)
                    continue;
                candidate.push_back(participant);
                std::sort(candidate.begin(), candidate.end());
                candidate.erase(
                    std::unique(candidate.begin(), candidate.end()),
                    candidate.end());
                if (candidate.size() <= cycle_capacity)
                    candidate_cohorts.insert(std::move(candidate));
            }

            for (const auto &candidate_indices : candidate_cohorts)
            {
                if (candidate_indices.empty() ||
                    std::all_of(
                        candidate_indices.begin(),
                        candidate_indices.end(),
                        [&](const std::size_t cycle_index)
                        { return full_cycle_economy[cycle_index].eligible(); }))
                {
                    continue;
                }

                std::uint64_t participant_entries = 0u;
                for (const std::size_t cycle_index : candidate_indices)
                {
                    participant_entries +=
                        participant_objective_entries(cycle_index);
                }
                if (participant_entries >
                    participant_admission_entry_limit)
                {
                    continue;
                }

                ++dependent_cohort_snapshot_builds;
                auto candidate =
                    build_cycle_subset_transaction(candidate_indices);
                if (candidate.empty() || !fits_wave_budget(candidate))
                    continue;
                const auto score = score_economy_transaction(candidate);
                if (!score.eligible())
                {
                    ++dependent_cohort_payoff_rejections;
                    continue;
                }

                const bool improves_best =
                    !dependent_cycle_cohort ||
                    score.projected_net_benefit_ns >
                        dependent_cycle_cohort->economy
                            .projected_net_benefit_ns ||
                    (score.projected_net_benefit_ns ==
                         dependent_cycle_cohort->economy
                             .projected_net_benefit_ns &&
                     score.projected_service_gain_ns >
                         dependent_cycle_cohort->economy
                             .projected_service_gain_ns) ||
                    (score.projected_net_benefit_ns ==
                         dependent_cycle_cohort->economy
                             .projected_net_benefit_ns &&
                     score.projected_service_gain_ns ==
                         dependent_cycle_cohort->economy
                             .projected_service_gain_ns &&
                     candidate_indices <
                         dependent_cycle_cohort->cycle_indices);
                if (!improves_best)
                    continue;
                dependent_cycle_cohort = DependentCycleCohort{
                    .cycle_indices = candidate_indices,
                    .transaction = std::move(candidate),
                    .economy = score,
                    .participant_entries = static_cast<std::uint32_t>(
                        participant_entries),
                };
            }
            dependent_cohort_candidates = static_cast<std::uint64_t>(
                candidate_cohorts.size());

            if (dependent_cycle_cohort)
            {
                /* Joint policy eligibility is typed at the cohort boundary.
                 * Add its prerequisite identities to cycle admission exactly
                 * once so host evidence never calls an admitted dependency an
                 * individual-policy rejection. */
                for (const std::size_t cycle_index :
                     dependent_cycle_cohort->cycle_indices)
                {
                    if (std::find(
                            cycle_order.begin(),
                            cycle_order.end(),
                            cycle_index) == cycle_order.end())
                    {
                        cycle_order.push_back(cycle_index);
                    }
                }
            }
        }

        /*
         * Dynamic has two independent placement objectives: improve tier
         * residency and reduce participant makespan inside each apportioned
         * tier. Pure net-benefit ordering can permanently starve the second
         * objective when a bounded wave is continually replenished with
         * higher-valued tier exchanges. Give each objective one admission
         * opportunity: try the best tier-advancing cycle first, then try
         * profitable participant candidates until one is admitted. Immediately
         * after that single reservation succeeds, restore the remaining cycles
         * to descending marginal net-benefit order. This prevents the fairness
         * rule from monopolizing four of five production slots while retaining
         * the search past a participant candidate that conflicts with the exact
         * per-layer shadow-slot BOM.
         */
        ParticipantLaneReservation participant_reservation;
        const std::vector<std::size_t> economic_cycle_order = cycle_order;
        if (config_.max_concurrent_cycles >= 2u &&
            cycle_order.size() >= 2u)
        {
            const bool tier_axis_available = std::any_of(
                cycle_order.begin(),
                cycle_order.end(),
                [&](const std::size_t cycle_index)
                {
                    return advancesTierResidency(migrationCycleAxis(
                        full_transaction,
                        full_transaction.migration_cycles[cycle_index]));
                });
            const bool participant_axis_available = std::any_of(
                cycle_order.begin(),
                cycle_order.end(),
                [&](const std::size_t cycle_index)
                {
                    return advancesParticipantPlacement(migrationCycleAxis(
                        full_transaction,
                        full_transaction.migration_cycles[cycle_index]));
                });
            if (tier_axis_available && participant_axis_available)
            {
                participant_reservation = ParticipantLaneReservation::required();
                const auto first_tier_it = std::find_if(
                    cycle_order.begin(),
                    cycle_order.end(),
                    [&](const std::size_t cycle_index)
                    {
                        return advancesTierResidency(migrationCycleAxis(
                            full_transaction,
                            full_transaction.migration_cycles[cycle_index]));
                    });
                if (first_tier_it == cycle_order.end())
                {
                    throw std::logic_error(
                        "ExpertOverlay two-axis scheduler lost its tier candidate");
                }
                const std::size_t first_cycle = *first_tier_it;
                std::vector<std::size_t> axis_balanced_order;
                axis_balanced_order.reserve(cycle_order.size());
                axis_balanced_order.push_back(first_cycle);

                /*
                 * Keep participant candidates consecutive only until the first
                 * one is admitted. The admission loop below then restores the
                 * untouched tail to `economic_cycle_order`.
                 */
                for (const std::size_t cycle_index : cycle_order)
                {
                    if (cycle_index == first_cycle)
                        continue;
                    const auto axis = migrationCycleAxis(
                        full_transaction,
                        full_transaction.migration_cycles[cycle_index]);
                    if (advancesParticipantPlacement(axis))
                        axis_balanced_order.push_back(cycle_index);
                }
                for (const std::size_t cycle_index : cycle_order)
                {
                    if (std::find(
                            axis_balanced_order.begin(),
                            axis_balanced_order.end(),
                            cycle_index) == axis_balanced_order.end())
                    {
                        axis_balanced_order.push_back(cycle_index);
                    }
                }
                cycle_order = std::move(axis_balanced_order);
            }
        }

        MigrationCycleAxisCounts eligible_axis_counts;
        if (single_cycle_arbitration_summary)
        {
            eligible_axis_counts =
                single_cycle_arbitration_summary->policy_eligible_axes;
        }
        else
        {
            for (const std::size_t cycle_index : cycle_order)
            {
                eligible_axis_counts.add(migrationCycleAxis(
                    full_transaction,
                    full_transaction.migration_cycles[cycle_index]));
            }
        }
        const auto record_axis_admission = [&]
            (MoEOverlayResidencyTransaction &admitted,
             const std::vector<std::size_t> &admitted_candidate_indices,
             bool capacity_bounded,
             bool policy_bounded,
             std::uint64_t dependent_payoff_rejections,
             std::size_t capacity_rejected_cycles,
             std::size_t participant_axis_budget_rejected_cycles)
        {
            MigrationCycleAxisCounts admitted_axis_counts;
            for (const auto &cycle : admitted.migration_cycles)
            {
                admitted_axis_counts.add(
                    migrationCycleAxis(admitted, cycle));
            }
            MigrationCycleAxisCounts admitted_candidate_axis_counts;
            for (const std::size_t cycle_index :
                 admitted_candidate_indices)
            {
                if (cycle_index >=
                    full_transaction.migration_cycles.size())
                {
                    throw std::logic_error(
                        "ExpertOverlay cycle-admission evidence references an unknown candidate cycle");
                }
                admitted_candidate_axis_counts.add(migrationCycleAxis(
                    full_transaction,
                    full_transaction.migration_cycles[cycle_index]));
            }

            const auto candidate_cycles = single_cycle_arbitration_summary
                ? single_cycle_arbitration_summary->candidate_cycles
                : static_cast<std::uint64_t>(
                      full_transaction.migration_cycles.size());
            const auto policy_eligible_cycles =
                single_cycle_arbitration_summary
                    ? single_cycle_arbitration_summary
                          ->policy_eligible_cycles
                    : static_cast<std::uint64_t>(cycle_order.size());
            const auto individual_policy_rejected_cycles =
                single_cycle_arbitration_summary
                    ? single_cycle_arbitration_summary
                          ->individual_policy_rejected_cycles
                    : candidate_cycles - policy_eligible_cycles;
            const auto effective_capacity_rejected_cycles =
                single_cycle_arbitration_summary
                    ? single_cycle_arbitration_summary
                          ->capacity_rejected_cycles
                    : static_cast<std::uint64_t>(
                          capacity_rejected_cycles);
            const auto effective_participant_budget_rejections =
                single_cycle_arbitration_summary
                    ? single_cycle_arbitration_summary
                          ->participant_axis_budget_rejected_cycles
                    : static_cast<std::uint64_t>(
                          participant_axis_budget_rejected_cycles);
            const bool effective_capacity_bounded =
                single_cycle_arbitration_summary
                    ? effective_capacity_rejected_cycles != 0u
                    : capacity_bounded;
            const bool effective_policy_bounded =
                single_cycle_arbitration_summary
                    ? individual_policy_rejected_cycles != 0u ||
                          effective_participant_budget_rejections != 0u
                    : policy_bounded;

            if (eligible_axis_counts.total() != policy_eligible_cycles ||
                admitted_axis_counts.total() !=
                    admitted.migration_cycles.size() ||
                admitted_candidate_axis_counts.total() !=
                    admitted_candidate_indices.size())
            {
                throw std::logic_error(
                    "ExpertOverlay cycle-axis classification is not total and exclusive");
            }

            const auto typed_axis_counts = [](
                                                const MigrationCycleAxisCounts
                                                    &counts)
            {
                return MoEOptimizationCycleAxisCounts{
                    .tier_residency = static_cast<std::uint64_t>(
                        counts.tier_residency),
                    .participant_placement = static_cast<std::uint64_t>(
                        counts.participant_placement),
                    .combined = static_cast<std::uint64_t>(counts.combined),
                };
            };
            MoEOptimizationHostMovementAdmission admission{
                .authority = MoEOptimizationAuthority::Host,
                .transaction = admitted.candidate->epoch,
                .candidate_epoch = admitted.candidate->epoch,
                .cycle_capacity_kind =
                    config_.max_concurrent_cycles == 0u
                        ? MoEOptimizationCycleCapacityKind::Unbounded
                        : MoEOptimizationCycleCapacityKind::Bounded,
                .maximum_concurrent_cycles = static_cast<std::uint64_t>(
                    config_.max_concurrent_cycles),
                .candidate_cycles = candidate_cycles,
                .policy_eligible_cycles = policy_eligible_cycles,
                .policy_eligible_axes =
                    typed_axis_counts(eligible_axis_counts),
                .admitted_candidate_cycles =
                    static_cast<std::uint64_t>(
                        admitted_candidate_indices.size()),
                .admitted_candidate_axes =
                    typed_axis_counts(admitted_candidate_axis_counts),
                .admitted_physical_cycles = static_cast<std::uint64_t>(
                    admitted.migration_cycles.size()),
                .admitted_physical_axes =
                    typed_axis_counts(admitted_axis_counts),
                .individual_policy_rejected_cycles =
                    individual_policy_rejected_cycles,
                .dependent_payoff_rejected_cycles =
                    dependent_payoff_rejections,
                .capacity_rejected_cycles =
                    effective_capacity_rejected_cycles,
                .participant_axis_budget_rejected_cycles =
                    effective_participant_budget_rejections,
                .dependent_cohort_candidates =
                    dependent_cohort_candidates,
                .dependent_cohort_payoff_rejections =
                    dependent_cohort_payoff_rejections,
                .physical_cycle_recomposition =
                    admitted.migration_cycles.size() !=
                    admitted_candidate_indices.size(),
                .capacity_bounded = effective_capacity_bounded,
                .policy_bounded = effective_policy_bounded,
            };
            if (!admission.valid())
            {
                std::ostringstream detail;
                detail
                    << "ExpertOverlay host cycle-admission proof is internally inconsistent"
                    << ": candidate_cycles=" << admission.candidate_cycles
                    << " policy_eligible_cycles="
                    << admission.policy_eligible_cycles
                    << " admitted_candidate_cycles="
                    << admission.admitted_candidate_cycles
                    << " admitted_physical_cycles="
                    << admission.admitted_physical_cycles
                    << " individual_policy_rejected_cycles="
                    << admission.individual_policy_rejected_cycles
                    << " dependent_payoff_rejected_cycles="
                    << admission.dependent_payoff_rejected_cycles
                    << " capacity_rejected_cycles="
                    << admission.capacity_rejected_cycles
                    << " participant_axis_budget_rejected_cycles="
                    << admission.participant_axis_budget_rejected_cycles
                    << " dependent_cohort_payoff_rejections="
                    << dependent_cohort_payoff_rejections
                    << " maximum_concurrent_cycles="
                    << admission.maximum_concurrent_cycles;
                throw std::logic_error(detail.str());
            }
            admitted.host_admission = admission;

            if (single_cycle_arbitration_summary &&
                effective_capacity_rejected_cycles != 0u)
            {
                capacity_bounded_proposals_.fetch_add(
                    1u, std::memory_order_relaxed);
                target_cycles_omitted_.fetch_add(
                    single_cycle_arbitration_summary
                        ->target_cycles_omitted,
                    std::memory_order_relaxed);
                target_migrations_omitted_.fetch_add(
                    single_cycle_arbitration_summary
                        ->target_migrations_omitted,
                    std::memory_order_relaxed);
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    "capacity_bounded_proposals",
                    1.0,
                    "maintenance",
                    config_.perf_device,
                    {{"policy_eligible_cycles",
                      std::to_string(policy_eligible_cycles)},
                     {"admitted_cycles",
                      std::to_string(
                          admitted.migration_cycles.size())},
                     {"capacity_omitted_cycles",
                      std::to_string(
                          effective_capacity_rejected_cycles)},
                     {"max_concurrent_cycles",
                      std::to_string(
                          config_.max_concurrent_cycles)},
                     {"selection_policy",
                      "single_cycle_axis_arbitration"}});
            }

            /*
             * Candidate cycles and physical transfer cycles are deliberately
             * distinct coordinate systems. Applying several overlapping
             * multi-tier candidates to one placement can cause
             * build_transaction() to merge or split the final closed transfer
             * cycles. Rejection accounting belongs to the original candidate
             * set; transport capacity belongs to the recomposed physical set.
             * Publishing both prevents a valid recomposition from appearing
             * as an unclassified policy/capacity rejection.
             */
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "cycle_axis_admission",
                1.0,
                "maintenance",
                config_.perf_device,
                {
                    {"eligible_tier_residency_cycles",
                     std::to_string(
                         eligible_axis_counts.tier_residency)},
                    {"eligible_participant_placement_cycles",
                     std::to_string(
                         eligible_axis_counts.participant_placement)},
                    {"eligible_combined_cycles",
                     std::to_string(eligible_axis_counts.combined)},
                    {"admitted_tier_residency_cycles",
                     std::to_string(
                         admitted_axis_counts.tier_residency)},
                    {"admitted_participant_placement_cycles",
                     std::to_string(
                         admitted_axis_counts.participant_placement)},
                    {"admitted_combined_cycles",
                     std::to_string(admitted_axis_counts.combined)},
                    {"admitted_candidate_tier_residency_cycles",
                     std::to_string(
                         admitted_candidate_axis_counts.tier_residency)},
                    {"admitted_candidate_participant_placement_cycles",
                     std::to_string(
                         admitted_candidate_axis_counts
                             .participant_placement)},
                    {"admitted_candidate_combined_cycles",
                     std::to_string(
                         admitted_candidate_axis_counts.combined)},
                    {"candidate_cycles",
                     std::to_string(candidate_cycles)},
                    {"policy_eligible_cycles",
                     std::to_string(policy_eligible_cycles)},
                    {"admitted_cycles",
                     std::to_string(admitted.migration_cycles.size())},
                    {"admitted_candidate_cycles",
                     std::to_string(
                         admitted_candidate_indices.size())},
                    {"physical_cycle_recomposition",
                     admitted.migration_cycles.size() !=
                             admitted_candidate_indices.size()
                         ? "true"
                         : "false"},
                    {"individual_policy_rejected_cycles",
                     std::to_string(
                         individual_policy_rejected_cycles)},
                    {"dependent_payoff_rejected_cycles",
                     std::to_string(dependent_payoff_rejections)},
                    {"capacity_rejected_cycles",
                     std::to_string(
                         effective_capacity_rejected_cycles)},
                    {"participant_axis_budget_rejected_cycles",
                     std::to_string(
                         effective_participant_budget_rejections)},
                    {"dependent_cohort_candidates",
                     std::to_string(dependent_cohort_candidates)},
                    {"dependent_cohort_payoff_rejections",
                     std::to_string(
                         dependent_cohort_payoff_rejections)},
                    {"independent_axis_reservation_active",
                     participant_reservation.active() ? "true" : "false"},
                    {"independent_axis_reserved_participant_candidates",
                     std::to_string(
                         participant_reservation.reservedCandidates())},
                    {"capacity_bounded",
                     effective_capacity_bounded ? "true" : "false"},
                    {"policy_bounded",
                     effective_policy_bounded ? "true" : "false"},
                });
        };

        if (cycle_order.size() ==
                full_transaction.migration_cycles.size() &&
            participant_changes.size() <=
                participant_admission_entry_limit &&
            fits_wave_budget(full_transaction) &&
            (!config_.migration_economy_policy ||
             score_economy_transaction(full_transaction).eligible()))
        {
            record_axis_admission(
                full_transaction,
                cycle_order,
                false,
                false,
                0u,
                0u,
                0u);
            return finalize_economy(std::move(full_transaction));
        }

        std::vector<std::size_t> selected_cycles;
        std::optional<MoEOverlayResidencyTransaction> bounded;
        std::optional<CycleEconomyScore> bounded_economy_score;
        bool economy_limited =
            dependent_cohort_payoff_rejections != 0u &&
            !dependent_cycle_cohort;
        std::set<std::size_t>
            dependent_payoff_rejected_cycle_indices;
        std::uint64_t bounded_snapshot_builds =
            dependent_cohort_snapshot_builds;
        std::set<std::size_t> capacity_rejected_cycle_indices;
        std::set<std::size_t>
            participant_axis_budget_rejected_cycle_indices;
        std::uint32_t selected_participant_objective_entries = 0u;
        const bool dependent_cohort_seeded =
            dependent_cycle_cohort.has_value();
        if (dependent_cycle_cohort)
        {
            selected_cycles = dependent_cycle_cohort->cycle_indices;
            bounded = std::move(dependent_cycle_cohort->transaction);
            bounded_economy_score = dependent_cycle_cohort->economy;
            selected_participant_objective_entries =
                dependent_cycle_cohort->participant_entries;
            // Seeded work has already passed joint economy and capacity gates.
            // Observe it before skipping its indices in the ordinary loop.
            for (const auto cycle_index : selected_cycles)
                participant_reservation.recordAdmission(migrationCycleAxis(
                    full_transaction, full_transaction.migration_cycles[cycle_index]));
            if (participant_reservation.fulfilled())
                cycle_order = economic_cycle_order;
        }

        /**
         * Classify the untouched tail when the physical wave is full.
         * Seeded indices can occur later in either ordering, but they remain
         * admitted; both capacity exits must use this same exclusion rule.
         */
        const auto rejectUnadmittedTailForCapacity = [&](std::size_t begin)
        {
            for (std::size_t remaining = begin; remaining < cycle_order.size(); ++remaining)
            {
                const auto cycle_index = cycle_order[remaining];
                if (std::find(selected_cycles.begin(), selected_cycles.end(), cycle_index) ==
                    selected_cycles.end())
                    capacity_rejected_cycle_indices.insert(cycle_index);
            }
        };
        for (std::size_t order_position = 0u;
             order_position < cycle_order.size();
             ++order_position)
        {
            const std::size_t cycle_index = cycle_order[order_position];
            if (std::find(
                    selected_cycles.begin(),
                    selected_cycles.end(),
                    cycle_index) != selected_cycles.end())
            {
                continue;
            }
            if (config_.max_concurrent_cycles != 0u && bounded &&
                bounded->migration_cycles.size() >=
                    config_.max_concurrent_cycles)
            {
                rejectUnadmittedTailForCapacity(order_position);
                break;
            }
            const std::uint32_t cycle_participant_entries =
                participant_objective_entries(cycle_index);
            if (cycle_participant_entries >
                    participant_admission_entry_limit -
                        std::min(
                            participant_admission_entry_limit,
                            selected_participant_objective_entries))
            {
                participant_axis_budget_rejected_cycle_indices.insert(
                    cycle_index);
                continue;
            }
            auto trial_cycles = selected_cycles;
            trial_cycles.push_back(cycle_index);

            /*
             * Start from the published tier and participant state, then apply
             * exactly the selected closed cycles. Participant ownership is an
             * explicit candidate axis: deriving it again from cold-start order
             * would erase a same-tier skew correction or invent extra copies.
             */
            ++bounded_snapshot_builds;
            auto trial = build_cycle_subset_transaction(trial_cycles);
            if (trial.empty())
                continue;
            if (!fits_wave_budget(trial))
            {
                capacity_rejected_cycle_indices.insert(cycle_index);
                continue;
            }
            std::optional<CycleEconomyScore> trial_economy_score;
            if (config_.migration_economy_policy)
            {
                trial_economy_score =
                    score_economy_transaction(trial);
                const auto &policy =
                    *config_.migration_economy_policy;
                WideCost required_net_benefit =
                    bounded_economy_score
                        ? bounded_economy_score
                              ->projected_net_benefit_ns
                        : 0;
                checkedAddWide(
                    &required_net_benefit,
                    policy.minimum_net_benefit_ns,
                    "bounded transaction marginal payoff threshold");
                if (!trial_economy_score->residency_eligible ||
                    !trial_economy_score->payoffEligible() ||
                    trial_economy_score->projected_net_benefit_ns <=
                        required_net_benefit)
                {
                    /*
                     * A cycle can be profitable in the complete target yet
                     * not profitable with the cycles admitted so far. Skip it
                     * without publishing a dependent or gratuitous move. A
                     * prerequisite tier publication can make it independently
                     * admissible in the next histogram epoch.
                     */
                    economy_limited = true;
                    dependent_payoff_rejected_cycle_indices.insert(
                        cycle_index);
                    continue;
                }
            }
            selected_cycles = std::move(trial_cycles);
            bounded = std::move(trial);
            bounded_economy_score = std::move(trial_economy_score);
            selected_participant_objective_entries +=
                cycle_participant_entries;

            const auto admitted_axis = migrationCycleAxis(
                full_transaction,
                full_transaction.migration_cycles[cycle_index]);
            if (participant_reservation.recordAdmission(admitted_axis) ==
                ParticipantLaneReservation::Transition::Fulfilled)
            {
                /*
                 * The fairness reservation has fulfilled its complete contract.
                 * Keep the already attempted prefix stable for rejection
                 * accounting, then restore every untouched candidate to the
                 * original measured-economy order. No extra transaction build
                 * or owner-map scan is needed on this host-side hot path.
                 */
                std::vector<std::size_t> restored_order;
                restored_order.reserve(cycle_order.size());
                restored_order.insert(
                    restored_order.end(),
                    cycle_order.begin(),
                    cycle_order.begin() +
                        static_cast<std::ptrdiff_t>(order_position + 1u));
                for (const std::size_t economic_index :
                     economic_cycle_order)
                {
                    if (std::find(
                            restored_order.begin(),
                            restored_order.end(),
                            economic_index) == restored_order.end())
                    {
                        restored_order.push_back(economic_index);
                    }
                }
                cycle_order = std::move(restored_order);
            }

            /*
             * Cycles are already ordered by descending economic benefit. Once
             * the explicit cycle budget is full, every later trial would add
             * one more cycle and must fail the same budget check. Stopping here
             * avoids rebuilding the complete layer/expert owner map for every
             * omitted cycle; on large MoE models that otherwise turns a
             * one-cycle maintenance wave into seconds of foreground CPU work.
             */
            if (config_.max_concurrent_cycles != 0 &&
                bounded->migration_cycles.size() >=
                    config_.max_concurrent_cycles)
            {
                /*
                 * Every unexamined policy-eligible cycle is omitted solely
                 * because the concurrent pool is now full.  Preserve those
                 * identities separately from marginal-payoff rejections so
                 * capacity telemetry never mislabels an economical decision.
                 */
                rejectUnadmittedTailForCapacity(order_position + 1u);
                break;
            }
        }

        bounded_candidate_snapshot_builds_.fetch_add(
            bounded_snapshot_builds, std::memory_order_relaxed);

        if (!bounded)
        {
            if ((cycle_order.empty() || economy_limited) &&
                config_.migration_economy_policy)
            {
                MoERoutedExpertPlacementPlan unchanged_plan =
                    *full_candidate->placement_plan;
                unchanged_plan.placements =
                    previous->placement_plan->placements;
                auto unchanged = buildSnapshot(
                    previous->epoch + 1,
                    std::move(unchanged_plan),
                    config_.model_metadata,
                    &previous->owner_map);
                return finalize_economy(
                    build_transaction(std::move(unchanged), {}));
            }
            throw std::runtime_error(
                "ExpertOverlay cannot express one capacity-preserving migration cycle within the configured shadow-slot and concurrency BOM");
        }

        if (dependent_cohort_seeded)
        {
            MigrationCycleAxisCounts cohort_axes;
            for (const std::size_t cycle_index :
                 dependent_cycle_cohort->cycle_indices)
            {
                cohort_axes.add(migrationCycleAxis(
                    full_transaction,
                    full_transaction.migration_cycles[cycle_index]));
            }
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "dependent_cycle_cohorts_admitted",
                1.0,
                "maintenance",
                config_.perf_device,
                {{"cycles",
                  std::to_string(
                      dependent_cycle_cohort->cycle_indices.size())},
                 {"tier_residency_cycles",
                  std::to_string(cohort_axes.tier_residency)},
                 {"participant_placement_cycles",
                  std::to_string(cohort_axes.participant_placement)},
                 {"combined_cycles",
                  std::to_string(cohort_axes.combined)},
                 {"projected_net_benefit_ns",
                  std::to_string(
                      dependent_cycle_cohort->economy
                          .projected_net_benefit_ns)}});
        }

        std::size_t capacity_omitted_migrations = 0u;
        for (const std::size_t cycle_index :
             capacity_rejected_cycle_indices)
        {
            capacity_omitted_migrations +=
                full_transaction.migration_cycles[cycle_index]
                    .migration_indices.size();
        }
        const std::uint64_t capacity_omitted_cycles =
            static_cast<std::uint64_t>(
                capacity_rejected_cycle_indices.size());
        const std::uint64_t capacity_omitted_migration_count =
            static_cast<std::uint64_t>(capacity_omitted_migrations);
        if (capacity_omitted_cycles != 0u)
        {
            capacity_bounded_proposals_.fetch_add(
                1, std::memory_order_relaxed);
            target_migrations_omitted_.fetch_add(
                capacity_omitted_migration_count,
                std::memory_order_relaxed);
            target_cycles_omitted_.fetch_add(
                capacity_omitted_cycles,
                std::memory_order_relaxed);
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "capacity_bounded_proposals",
                1.0,
                "maintenance",
                config_.perf_device,
                {{"policy_eligible_migrations",
                  std::to_string(
                      bounded->migrations.size() +
                      capacity_omitted_migration_count)},
                 {"admitted_migrations",
                  std::to_string(bounded->migrations.size())},
                 {"policy_eligible_cycles",
                  std::to_string(cycle_order.size())},
                 {"admitted_cycles",
                  std::to_string(bounded->migration_cycles.size())},
                 {"capacity_omitted_migrations",
                  std::to_string(capacity_omitted_migration_count)},
                 {"capacity_omitted_cycles",
                  std::to_string(capacity_omitted_cycles)},
                 {"shadow_slots_per_endpoint_layer",
                  std::to_string(
                      config_.shadow_slots_per_endpoint_layer)},
                 {"max_concurrent_cycles",
                  std::to_string(config_.max_concurrent_cycles)}});
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "bounded_candidate_snapshot_builds",
                static_cast<double>(bounded_snapshot_builds),
                "maintenance",
                config_.perf_device,
                {{"policy_eligible_cycles",
                  std::to_string(cycle_order.size())},
                 {"admitted_cycles",
                  std::to_string(bounded->migration_cycles.size())},
                 {"max_concurrent_cycles",
                  std::to_string(config_.max_concurrent_cycles)}});
        }
        if (!dependent_payoff_rejected_cycle_indices.empty())
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "dependent_payoff_rejections",
                static_cast<double>(
                    dependent_payoff_rejected_cycle_indices.size()),
                "maintenance",
                config_.perf_device,
                {{"histogram_generation",
                  std::to_string(transaction.histogram_generation)}});
        }
        if (dependent_cohort_payoff_rejections != 0u)
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "dependent_cohort_payoff_rejections",
                static_cast<double>(
                    dependent_cohort_payoff_rejections),
                "maintenance",
                config_.perf_device,
                {{"histogram_generation",
                  std::to_string(transaction.histogram_generation)},
                 {"candidate_cohorts",
                  std::to_string(dependent_cohort_candidates)}});
        }
        if (!participant_axis_budget_rejected_cycle_indices.empty())
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "participant_axis_budget_rejected_cycles",
                static_cast<double>(
                    participant_axis_budget_rejected_cycle_indices.size()),
                "maintenance",
                config_.perf_device,
                {{"histogram_generation",
                  std::to_string(transaction.histogram_generation)},
                 {"participant_admission_entry_limit",
                  std::to_string(participant_admission_entry_limit)}});
        }
        const bool policy_bounded =
            cycle_order.size() !=
                full_transaction.migration_cycles.size() ||
            !dependent_payoff_rejected_cycle_indices.empty() ||
            !participant_axis_budget_rejected_cycle_indices.empty();
        std::set<std::size_t> classified_cycle_indices(
            selected_cycles.begin(), selected_cycles.end());
        const auto add_disjoint_classification = [&]
            (const std::set<std::size_t> &indices,
             const char *classification)
        {
            for (const std::size_t cycle_index : indices)
            {
                if (!classified_cycle_indices.insert(cycle_index).second)
                {
                    throw std::logic_error(
                        std::string("ExpertOverlay cycle belongs to more than one host-admission classification: ") +
                        classification);
                }
            }
        };
        add_disjoint_classification(
            dependent_payoff_rejected_cycle_indices,
            "dependent payoff");
        add_disjoint_classification(
            capacity_rejected_cycle_indices,
            "capacity");
        add_disjoint_classification(
            participant_axis_budget_rejected_cycle_indices,
            "participant axis budget");
        if (classified_cycle_indices.size() != cycle_order.size())
        {
            throw std::logic_error(
                "ExpertOverlay host admission did not classify every policy-eligible cycle exactly once");
        }
        record_axis_admission(
            *bounded,
            selected_cycles,
            capacity_omitted_cycles != 0u,
            policy_bounded,
            dependent_payoff_rejected_cycle_indices.size(),
            capacity_rejected_cycle_indices.size(),
            participant_axis_budget_rejected_cycle_indices.size());
        return finalize_economy(std::move(*bounded));
    }

    MoEOverlayAuthoritativeResidencyPlan
    MoEOverlayResidencyAuthority::exportAuthoritativeResidencyPlan(
        const MoEOverlayResidencyTransaction &transaction) const
    {
        if (!transaction.valid() ||
            transaction.purpose !=
                MoEOverlayResidencyTransactionPurpose::PlacementChange ||
            !transaction.histogram_window ||
            transaction.histogram_window->first_model_layer != config_.model_metadata.first_model_layer ||
            transaction.histogram_window->num_layers != config_.model_metadata.num_layers ||
            transaction.histogram_window->num_experts != config_.model_metadata.num_experts)
        {
            throw std::invalid_argument(
                "Only a valid live ExpertOverlay placement transaction can be published as an authoritative plan");
        }
        const auto current = snapshot();
        if (!current || current.get() != transaction.previous.get() ||
            current.owner_before(transaction.previous) || transaction.previous.owner_before(current))
            throw std::logic_error("Authoritative export requires this authority's exact current source");
        if (transaction.previous->epoch != transaction.expected_epoch ||
            transaction.candidate->epoch != transaction.expected_epoch + 1u)
        {
            throw std::logic_error(
                "ExpertOverlay authoritative plan has invalid epoch progression");
        }

        MoEOverlayAuthoritativeResidencyPlan plan{
            .expected_epoch = transaction.expected_epoch,
            .num_layers = config_.model_metadata.num_layers,
            .num_experts = config_.model_metadata.num_experts,
            .histogram_window = transaction.histogram_window,
        };
        const size_t entry_count =
            static_cast<size_t>(plan.num_layers) *
            static_cast<size_t>(plan.num_experts);
        plan.entries.resize(entry_count);

        std::map<std::pair<int, int>, const MoEOverlayTierMigration *>
            migration_by_coordinate;
        for (const auto &migration : transaction.migrations)
        {
            if (!migration_by_coordinate
                     .emplace(
                         std::pair{migration.layer_idx, migration.expert_id},
                         &migration)
                     .second)
            {
                throw std::logic_error(
                    "ExpertOverlay authoritative transaction repeats a migration coordinate");
            }
        }

        size_t changed_entries = 0;
        for (int row = 0; row < plan.num_layers; ++row)
        {
            const int layer_idx = plan.firstModelLayer() + row;
            const auto placement = std::find_if(
                transaction.candidate->placement_plan->placements.begin(),
                transaction.candidate->placement_plan->placements.end(),
                [&](const auto &candidate)
                { return candidate.layer == layer_idx; });
            if (placement ==
                    transaction.candidate->placement_plan->placements.end() ||
                placement->routed_expert_tier.size() !=
                    static_cast<size_t>(plan.num_experts))
            {
                throw std::logic_error(
                    "ExpertOverlay authoritative candidate lacks complete layer placement geometry");
            }
            for (int expert_id = 0; expert_id < plan.num_experts; ++expert_id)
            {
                const auto *previous_owner =
                    transaction.previous->placementOwnerMap().placementFor(
                        layer_idx, expert_id);
                const auto *candidate_owner =
                    transaction.candidate->placementOwnerMap().placementFor(
                        layer_idx, expert_id);
                if (!previous_owner || !candidate_owner)
                {
                    throw std::logic_error(
                        "ExpertOverlay authoritative candidate has incomplete owner maps");
                }

                auto &entry = plan.entries[plan.offset(layer_idx, expert_id)];
                entry.candidate_tier_idx =
                    placement->routed_expert_tier[
                        static_cast<size_t>(expert_id)];
                const auto checked_target = MoEExpertPlacementTarget::fromPlacement(*candidate_owner);
                if (!checked_target)
                    throw std::logic_error("Authoritative candidate has an incomplete fixed target");
                entry.candidate_target = checked_target->isCohort()
                    ? MoEOverlayRouteTarget::cpuCohort(candidate_owner->tier_idx)
                    : MoEOverlayRouteTarget::whole(candidate_owner->owner_participant);

                const auto migration = migration_by_coordinate.find(
                    {layer_idx, expert_id});
                const bool snapshot_changed =
                    !(*previous_owner == *candidate_owner);
                if (migration == migration_by_coordinate.end())
                {
                    if (snapshot_changed)
                    {
                        throw std::logic_error(
                            "ExpertOverlay authoritative migration set does not cover its candidate snapshot delta");
                    }
                    continue;
                }
                if (!snapshot_changed ||
                    migration->second->destination.tier_idx !=
                        entry.candidate_tier_idx ||
                    !(migration->second->destination == *candidate_owner))
                {
                    throw std::logic_error(
                        "ExpertOverlay authoritative migration disagrees with its candidate snapshot");
                }
                entry.changed = true;
                entry.axis = migration->second->axis;
                entry.activation_count =
                    migration->second->activation_count;
                entry.estimated_weight_bytes =
                    migration->second->estimated_weight_bytes;
                ++changed_entries;
            }
        }
        if (changed_entries != transaction.migrations.size() || !plan.valid())
        {
            throw std::logic_error(
                "ExpertOverlay authoritative plan did not encode one exact dense transaction");
        }
        return plan;
    }

    MoEOverlayResidencyTransaction
    MoEOverlayResidencyAuthority::adoptAuthoritativeResidencyPlan(
        const MoEOverlayAuthoritativeResidencyPlan &plan)
    {
        requireHostDynamicPublicationAuthority(
            "Host ExpertOverlay authoritative-plan adoption");
        if (!migrationEnabled())
        {
            throw std::logic_error(
                "Only Dynamic ExpertOverlay maintenance may adopt an authoritative plan");
        }
        return reconstructAuthoritativeResidencyPlan(plan);
    }

    MoEOverlayResidencyTransaction
    MoEOverlayResidencyAuthority::reconstructAuthoritativeResidencyPlan(
        const MoEOverlayAuthoritativeResidencyPlan &plan)
    {
        if (!plan.valid() ||
            plan.num_layers != config_.model_metadata.num_layers ||
            plan.firstModelLayer() != config_.model_metadata.first_model_layer ||
            plan.num_experts != config_.model_metadata.num_experts)
        {
            throw std::invalid_argument(
                "Distributed ExpertOverlay authoritative plan has invalid model geometry");
        }
        const auto histogram_evidence =
            plan.histogram_window->validatedView();

        const auto previous = snapshot();
        if (!previous || !previous->valid() ||
            previous->epoch != plan.expected_epoch)
        {
            throw std::logic_error(
                "Distributed ExpertOverlay authoritative plan references a stale local epoch");
        }

        MoERoutedExpertPlacementPlan candidate_plan =
            *previous->placement_plan;
        MoELayeredExpertOwnership candidate_ownership =
            previous->layered_ownership;
        for (int row = 0; row < plan.num_layers; ++row)
        {
            const int layer_idx = plan.firstModelLayer() + row;
            const auto placement = std::find_if(
                candidate_plan.placements.begin(),
                candidate_plan.placements.end(),
                [&](const auto &candidate)
                { return candidate.layer == layer_idx; });
            if (placement == candidate_plan.placements.end() ||
                placement->routed_expert_tier.size() !=
                    static_cast<size_t>(plan.num_experts))
            {
                throw std::logic_error(
                    "Distributed ExpertOverlay local topology lacks authoritative layer geometry");
            }
            for (int expert_id = 0; expert_id < plan.num_experts; ++expert_id)
            {
                const auto &entry =
                    plan.entries[plan.offset(layer_idx, expert_id)];
                placement->routed_expert_tier[
                    static_cast<size_t>(expert_id)] =
                    entry.candidate_tier_idx;
                if (!previous->originalProjectionPreparation())
                {
                    if (!entry.candidate_target.isWhole())
                        throw std::invalid_argument("Whole source cannot adopt a CPU cohort tag");
                    candidate_ownership.assignOwner(layer_idx, expert_id, entry.candidate_target.word());
                }
            }
        }

        auto candidate = previous->originalProjectionPreparation()
            ? prepareCPUJointPlacementCandidate(previous, previous->epoch, candidate_plan.placements)
            : buildSnapshot(previous->epoch + 1u, std::move(candidate_plan),
                            config_.model_metadata, &previous->owner_map, &candidate_ownership);
        if (previous->originalProjectionPreparation())
        {
            std::vector<MoELayeredExpertOwnershipChange> changes;
            for (const auto &owner : candidate->placementOwnerMap().owners())
            {
                const auto &entry = plan.entries[plan.offset(owner.layer_idx, owner.expert_id)];
                if (!owner.shard_cohort && entry.candidate_target.isWhole() &&
                    owner.owner_participant != entry.candidate_target.word())
                    changes.push_back({.layer_idx = owner.layer_idx, .expert_id = owner.expert_id,
                        .previous_participant = owner.owner_participant,
                        .current_participant = entry.candidate_target.word()});
            }
            if (!changes.empty())
                candidate = prepareCPUJointPlacementCandidate(previous, previous->epoch,
                    candidate_plan.placements, changes);
        }
        // Authenticate every dense coordinate, including unchanged targets, against
        // the actual source-derived complete placement. A cohort tag is not membership.
        for (int row = 0; row < plan.num_layers; ++row)
            for (int expert = 0; expert < plan.num_experts; ++expert)
            {
                const int layer = plan.firstModelLayer() + row;
                const auto *owner = candidate->placementOwnerMap().placementFor(layer, expert);
                const auto target = owner ? MoEExpertPlacementTarget::fromPlacement(*owner) : std::nullopt;
                if (!target)
                    throw std::invalid_argument("Authoritative target lacks a complete original placement");
                const auto expected = target->isCohort()
                    ? MoEOverlayRouteTarget::cpuCohort(owner->tier_idx)
                    : MoEOverlayRouteTarget::whole(owner->owner_participant);
                const auto &entry = plan.entries[plan.offset(layer, expert)];
                if (entry.candidate_tier_idx != owner->tier_idx ||
                    entry.candidate_target.word() != expected.word())
                    throw std::invalid_argument("Authoritative target differs from the complete original source placement");
            }
        requireCapacityPreserving(*previous, *candidate);

        MoEOverlayResidencyTransaction transaction;
        transaction.expected_epoch = previous->epoch;
        transaction.histogram_generation =
            plan.histogram_window->generation;
        transaction.histogram_window = plan.histogram_window;
        transaction.previous = previous;
        transaction.candidate = std::move(candidate);

        for (int row = 0; row < plan.num_layers; ++row)
        {
            const int layer_idx = plan.firstModelLayer() + row;
            for (int expert_id = 0; expert_id < plan.num_experts; ++expert_id)
            {
                const auto &entry =
                    plan.entries[plan.offset(layer_idx, expert_id)];
                const auto *source = previous->placementOwnerMap().placementFor(
                    layer_idx, expert_id);
                const auto *destination =
                    transaction.candidate->placementOwnerMap().placementFor(
                        layer_idx, expert_id);
                if (!source || !destination)
                {
                    throw std::logic_error(
                        "Distributed ExpertOverlay authoritative plan resolved an incomplete owner map");
                }
                const bool changed =
                    !(*source == *destination);
                if (changed != entry.changed)
                {
                    throw std::logic_error(
                        "Distributed ExpertOverlay authoritative change mask disagrees with local topology");
                }
                if (!changed)
                    continue;

                const uint64_t expected_activation_count =
                    histogram_evidence.activationCount(
                        layer_idx, expert_id);
                if (entry.activation_count != expected_activation_count)
                {
                    std::ostringstream diagnostic;
                    diagnostic
                        << "Distributed ExpertOverlay authoritative activation count disagrees with its histogram"
                        << " layer=" << layer_idx
                        << " expert=" << expert_id
                        << " entry_count=" << entry.activation_count
                        << " histogram_count=" << expected_activation_count
                        << " histogram_generation="
                        << plan.histogram_window->generation
                        << " expected_epoch=" << plan.expected_epoch;
                    throw std::invalid_argument(diagnostic.str());
                }
                const int source_priority = tierPriority(
                    *previous->placement_plan, source->tier_idx);
                const int destination_priority = tierPriority(
                    *transaction.candidate->placement_plan,
                    destination->tier_idx);
                transaction.migrations.push_back({
                    .layer_idx = layer_idx,
                    .expert_id = expert_id,
                    .activation_count = entry.activation_count,
                    .estimated_weight_bytes =
                        entry.estimated_weight_bytes,
                    .direction =
                        destination_priority < source_priority
                            ? MoEOverlayTierMigrationDirection::Promotion
                            : (destination_priority > source_priority
                                   ? MoEOverlayTierMigrationDirection::Demotion
                                   : MoEOverlayTierMigrationDirection::
                                         SamePriority),
                    .axis = entry.axis,
                    .source = *source,
                    .destination = *destination,
                });
            }
        }
        transaction.migration_cycles = buildMoEOverlayMigrationCycles(
            transaction.migrations);
        transaction.shadow_requirements = buildMoEOverlayShadowRequirements(
            transaction.migrations);
        if (!transaction.valid())
        {
            throw std::logic_error(
                "Distributed ExpertOverlay authoritative plan reconstructed an invalid transaction");
        }
        checks_.fetch_add(1, std::memory_order_relaxed);
        return transaction;
    }

    MoEOverlayResidencyApplyResult MoEOverlayResidencyAuthority::beginApply(
        const MoEOverlayResidencyTransaction &transaction,
        IMoEOverlayResidencyTransport &transport)
    {
        requireHostDynamicPublicationAuthority(
            "Host ExpertOverlay residency apply");
        MoEOverlayResidencyApplyResult result;
        const bool static_policy = !migrationEnabled();
        const bool publishable_purpose =
            transaction.purpose ==
                MoEOverlayResidencyTransactionPurpose::PlacementChange ||
            transaction.purpose ==
                MoEOverlayResidencyTransactionPurpose::
                    PreparedContextRestoration;
        if (!transaction.valid() || !publishable_purpose)
        {
            result.status = MoEOverlayResidencyApplyStatus::Stale;
            result.error = transaction.purpose ==
                                   MoEOverlayResidencyTransactionPurpose::
                                       EconomyCalibration
                               ? "Economy calibration transactions cannot enter residency publication"
                               : "Residency transaction is structurally invalid or has a non-publishable purpose";
            stale_rejections_.fetch_add(1, std::memory_order_relaxed);
            return result;
        }

        std::lock_guard<std::mutex> lock(maintenance_mutex_);
        reapReadyAbortsLocked();
        std::string retirement_error;
        if (!reapReadyRetirementsLocked(&retirement_error))
        {
            result.status = MoEOverlayResidencyApplyStatus::RetirementFailed;
            result.error = retirement_error.empty()
                               ? "Previous ExpertOverlay epoch retirement fence failed"
                               : std::move(retirement_error);
            return result;
        }

        if (active_wave_)
        {
            result.status = MoEOverlayResidencyApplyStatus::Busy;
            result.error = "Another residency maintenance wave is active";
            busy_rejections_.fetch_add(1, std::memory_order_relaxed);
            return result;
        }

        /*
         * Two-bank participant storage cannot admit another candidate until
         * every rank has released and retired the previous epoch.  Returning
         * Deferred preserves the already-derived transaction for retry while
         * the maintenance worker continues polling the non-blocking global
         * lease fence.
         */
        if (!pending_retirements_.empty())
        {
            result.status = MoEOverlayResidencyApplyStatus::Deferred;
            result.error =
                "Previous ExpertOverlay epoch is awaiting all-rank lease drainage";
            const auto current =
                published_epoch_.load(std::memory_order_acquire);
            result.published_epoch =
                current && current->snapshot ? current->snapshot->epoch : 0;
            return result;
        }

        const auto current_state =
            published_epoch_.load(std::memory_order_acquire);
        const auto current = current_state ? current_state->snapshot : nullptr;
        if (!current_state || !current ||
            current.get() != transaction.previous.get() ||
            current->epoch != transaction.expected_epoch)
        {
            result.status = MoEOverlayResidencyApplyStatus::Stale;
            result.error = "Residency transaction was planned from a stale epoch";
            result.published_epoch = current ? current->epoch : 0;
            stale_rejections_.fetch_add(1, std::memory_order_relaxed);
            return result;
        }

        if (transaction.empty())
        {
            recordNoMovement(static_policy);
            result.status = static_policy
                                ? MoEOverlayResidencyApplyStatus::StaticNoMovement
                                : MoEOverlayResidencyApplyStatus::DynamicNoMovement;
            result.published_epoch = current->epoch;
            return result;
        }

        if (static_policy)
        {
            throw std::logic_error(
                "Static MoE overlay residency produced a non-empty migration wave");
        }

        MoEOverlayResidencyStageStart start;
        try
        {
            start = transport.beginStage(transaction);
        }
        catch (const std::exception &error)
        {
            start.status = MoEOverlayResidencyStageStartStatus::Failed;
            start.error = error.what();
        }
        catch (...)
        {
            start.status = MoEOverlayResidencyStageStartStatus::Failed;
            start.error = "Background residency transport threw a non-standard exception";
        }

        if (!start.valid())
        {
            /*
             * Even a malformed transport result may own already-fenced work.
             * Normalize every such owner through the abort queue before
             * reporting the structural failure.
             */
            abortAndRetainWaveLocked(std::move(start.wave), current->epoch);
            abortAndRetainWaveLocked(
                std::move(start.cleanup_wave),
                current->epoch);
            start.status = MoEOverlayResidencyStageStartStatus::Failed;
            if (start.error.empty())
                start.error = "Background residency transport returned invalid ownership";
        }

        if (start.status == MoEOverlayResidencyStageStartStatus::Deferred)
        {
            deferred_waves_.fetch_add(1, std::memory_order_relaxed);
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "migration_waves_deferred",
                1.0,
                "maintenance",
                config_.perf_device,
                baseTags(config_.initial_plan.residency_policy, current->epoch));
            result.status = MoEOverlayResidencyApplyStatus::Deferred;
            result.error = start.error.empty()
                               ? "Background shadow capacity is temporarily unavailable"
                               : std::move(start.error);
            result.published_epoch = current->epoch;
            return result;
        }

        if (start.status == MoEOverlayResidencyStageStartStatus::Failed)
        {
            abortAndRetainWaveLocked(
                std::move(start.cleanup_wave),
                current->epoch);
            stage_failures_.fetch_add(1, std::memory_order_relaxed);
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "migration_stage_failures",
                1.0,
                "maintenance",
                config_.perf_device,
                baseTags(config_.initial_plan.residency_policy, current->epoch));
            result.status = MoEOverlayResidencyApplyStatus::StageFailed;
            result.error = start.error.empty()
                               ? "Failed to enqueue background expert preparation"
                               : std::move(start.error);
            result.published_epoch = current->epoch;
            return result;
        }

        active_wave_ = std::make_unique<ActiveBackgroundWave>();
        active_wave_->transaction = transaction;
        active_wave_->previous = current_state;
        active_wave_->work = std::move(start.wave);
        background_waves_started_.fetch_add(1, std::memory_order_relaxed);
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "background_waves_started",
            1.0,
            "maintenance",
            config_.perf_device,
            baseTags(config_.initial_plan.residency_policy, current->epoch));

        result.status = MoEOverlayResidencyApplyStatus::Started;
        result.published_epoch = current->epoch;
        result.migration_count = transaction.migrations.size();
        return result;
    }

    MoEOverlayResidencyApplyResult
    MoEOverlayResidencyAuthority::advanceBackground()
    {
        requireHostDynamicPublicationAuthority(
            "Host ExpertOverlay background publication");
        std::lock_guard<std::mutex> lock(maintenance_mutex_);
        reapReadyAbortsLocked();
        std::string retirement_error;
        if (!reapReadyRetirementsLocked(&retirement_error))
        {
            const auto current =
                published_epoch_.load(std::memory_order_acquire);
            return {
                .status = MoEOverlayResidencyApplyStatus::RetirementFailed,
                .published_epoch = current && current->snapshot
                                       ? current->snapshot->epoch
                                       : 0,
                .error = retirement_error.empty()
                             ? "Previous ExpertOverlay epoch retirement fence failed"
                             : std::move(retirement_error),
            };
        }

        if (!active_wave_)
        {
            const auto current = snapshot();
            return {
                // The old bank may still own a distributed retirement vote.
                // Reporting Idle here used to let the sole progress worker
                // leave that vote for an arbitrarily expensive next proposal.
                .status = pending_retirements_.empty() && pending_aborts_.empty()
                              ? MoEOverlayResidencyApplyStatus::Idle
                              : MoEOverlayResidencyApplyStatus::Reclaiming,
                .published_epoch = current ? current->epoch : 0,
            };
        }

        const auto current_state =
            published_epoch_.load(std::memory_order_acquire);
        if (current_state != active_wave_->previous ||
            !current_state ||
            !current_state->snapshot ||
            current_state->snapshot.get() !=
                active_wave_->transaction.previous.get())
        {
            stale_rejections_.fetch_add(1, std::memory_order_relaxed);
            return failActiveWaveLocked(
                MoEOverlayResidencyApplyStatus::Stale,
                "Background residency wave was based on a stale epoch");
        }

        std::string transport_error;
        if (active_wave_->phase == ActiveBackgroundWave::Phase::Staging)
        {
            const auto progress = active_wave_->work->pollStage(&transport_error);
            if (progress == MoEOverlayResidencyWaveProgress::Pending)
            {
                return {
                    .status = MoEOverlayResidencyApplyStatus::Staging,
                    .published_epoch = current_state->snapshot->epoch,
                    .migration_count = active_wave_->transaction.migrations.size(),
                };
            }
            if (progress == MoEOverlayResidencyWaveProgress::Deferred)
            {
                deferred_waves_.fetch_add(1, std::memory_order_relaxed);
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    "migration_waves_deferred",
                    1.0,
                    "maintenance",
                    config_.perf_device,
                    baseTags(
                        config_.initial_plan.residency_policy,
                        current_state->snapshot->epoch));
                return failActiveWaveLocked(
                    MoEOverlayResidencyApplyStatus::Deferred,
                    transport_error.empty()
                        ? "Distributed ExpertOverlay stage was globally deferred"
                        : std::move(transport_error));
            }
            if (progress == MoEOverlayResidencyWaveProgress::Failed)
            {
                return failActiveWaveLocked(
                    MoEOverlayResidencyApplyStatus::StageFailed,
                    transport_error.empty()
                        ? "Background expert preparation or transfer failed"
                        : std::move(transport_error));
            }
            if (!active_wave_->work->beginPrepare(&transport_error))
            {
                return failActiveWaveLocked(
                    MoEOverlayResidencyApplyStatus::PreparationFailed,
                    transport_error.empty()
                        ? "Failed to enqueue inactive-bank preparation"
                        : std::move(transport_error));
            }

            active_wave_->phase = ActiveBackgroundWave::Phase::Preparing;
            return {
                .status = MoEOverlayResidencyApplyStatus::Preparing,
                .published_epoch = current_state->snapshot->epoch,
                .migration_count = active_wave_->transaction.migrations.size(),
            };
        }

        if (active_wave_->phase == ActiveBackgroundWave::Phase::Preparing)
        {
            const auto progress =
                active_wave_->work->pollPrepare(&transport_error);
            if (progress == MoEOverlayResidencyWaveProgress::Pending)
            {
                return {
                    .status = MoEOverlayResidencyApplyStatus::Preparing,
                    .published_epoch = current_state->snapshot->epoch,
                    .migration_count =
                        active_wave_->transaction.migrations.size(),
                };
            }
            if (progress != MoEOverlayResidencyWaveProgress::Ready)
            {
                return failActiveWaveLocked(
                    MoEOverlayResidencyApplyStatus::PreparationFailed,
                    transport_error.empty()
                        ? "Inactive expert bank preparation failed"
                        : std::move(transport_error));
            }

            auto expected_admission = PublishedEpochState::
                GraphSequenceAdmissionState::Open;
            if (!current_state->graph_sequence_admission.compare_exchange_strong(
                    expected_admission,
                    PublishedEpochState::GraphSequenceAdmissionState::
                        PublicationReserved,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire))
            {
                LOG_ERROR(
                    "ExpertOverlay prepared wave found a non-open graph-sequence publication gate");
                std::terminate();
            }
            publication_boundary_reservations_.fetch_add(
                1, std::memory_order_relaxed);
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "publication_boundary_reservations",
                1.0,
                "maintenance",
                config_.perf_device,
                baseTags(
                    config_.initial_plan.residency_policy,
                    current_state->snapshot->epoch));
            active_wave_->phase =
                ActiveBackgroundWave::Phase::AwaitingGraphSequenceBoundary;
        }

        if (active_wave_->phase == ActiveBackgroundWave::Phase::
                                      AwaitingGraphSequenceBoundary)
        {
            const uint64_t active_graph_sequences =
                current_state->active_graph_sequences.load(
                    std::memory_order_acquire);
            if (active_graph_sequences != 0)
            {
                if (!active_wave_->graph_sequence_drain_recorded)
                {
                    active_wave_->graph_sequence_drain_recorded = true;
                    publication_boundary_drains_.fetch_add(
                        1, std::memory_order_relaxed);
                    PerfStatsCollector::addCounter(
                        "moe_overlay_residency",
                        "publication_boundary_drains",
                        1.0,
                        "maintenance",
                        config_.perf_device,
                        {{"active_graph_sequences",
                          std::to_string(active_graph_sequences)},
                         {"epoch",
                          std::to_string(current_state->snapshot->epoch)}});
                }
                return {
                    .status = MoEOverlayResidencyApplyStatus::
                        AwaitingGraphSequenceBoundary,
                    .published_epoch = current_state->snapshot->epoch,
                    .migration_count =
                        active_wave_->transaction.migrations.size(),
                };
            }

            /*
             * Exact device-selected tickets may observe E+1 as soon as the
             * first selector flips. Publish the same immutable state into this
             * narrow lookup slot before submitting any selector work. Ordinary
             * host admission continues to read only `published_epoch_` (E).
             */
            active_wave_->candidate =
                std::make_shared<PublishedEpochState>(
                    active_wave_->transaction.candidate);
            std::shared_ptr<PublishedEpochState> no_candidate;
            if (!candidate_epoch_.compare_exchange_strong(
                    no_candidate,
                    active_wave_->candidate,
                    std::memory_order_release,
                    std::memory_order_acquire))
            {
                return failActiveWaveLocked(
                    MoEOverlayResidencyApplyStatus::PreparationFailed,
                    "Another prepared ExpertOverlay candidate already owns the exact-ticket slot");
            }

            if (!active_wave_->work->beginPublication(&transport_error))
            {
                /*
                 * A valid prepared wave promises a total publication submit.
                 * Once exact E+1 admission is visible, guessing whether any
                 * participant selector changed would be an unsafe rollback.
                 */
                LOG_ERROR(
                    "ExpertOverlay publication failed after candidate exact admission opened: "
                    << (transport_error.empty()
                            ? "publication submit returned no diagnostic"
                            : transport_error));
                std::terminate();
            }
            active_wave_->phase = ActiveBackgroundWave::Phase::Publishing;
            return {
                .status = MoEOverlayResidencyApplyStatus::Publishing,
                .published_epoch = current_state->snapshot->epoch,
                .migration_count = active_wave_->transaction.migrations.size(),
            };
        }

        const auto transaction = active_wave_->transaction;
        const auto publication_progress =
            active_wave_->work->pollPublication(&transport_error);
        if (publication_progress == MoEOverlayResidencyWaveProgress::Pending)
        {
            return {
                .status = MoEOverlayResidencyApplyStatus::Publishing,
                .published_epoch = current_state->snapshot->epoch,
                .migration_count = transaction.migrations.size(),
            };
        }
        if (publication_progress != MoEOverlayResidencyWaveProgress::Ready)
        {
            LOG_ERROR(
                "ExpertOverlay selector publication failed after its irreversible edge: "
                << (transport_error.empty()
                        ? "publication poll returned no diagnostic"
                        : transport_error));
            std::terminate();
        }

        auto next_state = active_wave_->candidate;
        if (!next_state || !next_state->snapshot ||
            next_state->snapshot.get() != transaction.candidate.get() ||
            candidate_epoch_.load(std::memory_order_acquire) != next_state)
        {
            LOG_ERROR(
                "ExpertOverlay prepared candidate identity changed during selector publication");
            std::terminate();
        }
        auto expected_state = active_wave_->previous;
        std::shared_ptr<PublishedEpochState> expected_retiring;
        if (!retiring_epoch_.compare_exchange_strong(
                expected_retiring,
                expected_state,
                std::memory_order_release,
                std::memory_order_acquire))
        {
            LOG_ERROR(
                "A previous ExpertOverlay epoch still owns the exact-ticket retirement slot after selector publication");
            std::terminate();
        }
        if (!published_epoch_.compare_exchange_strong(
                expected_state,
                next_state,
                std::memory_order_release,
                std::memory_order_acquire))
        {
            LOG_ERROR(
                "Published ExpertOverlay epoch changed after selector publication");
            std::terminate();
        }

        auto expected_candidate = next_state;
        if (!candidate_epoch_.compare_exchange_strong(
                expected_candidate,
                std::shared_ptr<PublishedEpochState>{},
                std::memory_order_release,
                std::memory_order_acquire))
        {
            LOG_ERROR(
                "ExpertOverlay exact-ticket candidate slot changed during public-floor publication");
            std::terminate();
        }

        /* Notify protocol wrappers only after the candidate is the live epoch. */
        active_wave_->work->markAuthorityPublished();

        /*
         * Wake graph admissions only after every selector and the public host
         * floor name the successor. Waiters retry through `published_epoch_`;
         * they can never receive the superseded state whose gate they slept on.
         */
        active_wave_->previous->graph_sequence_admission.store(
            PublishedEpochState::GraphSequenceAdmissionState::Superseded,
            std::memory_order_release);
        active_wave_->previous->graph_sequence_admission.notify_all();

        const uint64_t old_ticket_count =
            active_wave_->previous->active_tickets.load(std::memory_order_acquire);
        if (old_ticket_count != 0)
        {
            published_with_old_tickets_.fetch_add(1, std::memory_order_relaxed);
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "published_with_old_tickets",
                1.0,
                "maintenance",
                config_.perf_device,
                baseTags(
                    config_.initial_plan.residency_policy,
                    transaction.candidate->epoch));
        }

        /* Ownership changes only after the complete inactive bank is visible. */
        if (transaction.candidate->jointOwnerMap())
            config_.histogram->updatePlacementOwnership(std::shared_ptr<const MoEExpertOwnerMap>(
                transaction.candidate, &transaction.candidate->placementOwnerMap()));
        else
            config_.histogram->updateOwnership(transaction.candidate->layered_ownership);
        recordCommitted(transaction);

        auto retirement = std::make_unique<PendingRetirement>();
        retirement->previous = std::move(active_wave_->previous);
        retirement->work = std::move(active_wave_->work);
        pending_retirements_.push_back(std::move(retirement));
        active_wave_.reset();
        if (!reapReadyRetirementsLocked(&retirement_error))
        {
            return {
                .status = MoEOverlayResidencyApplyStatus::RetirementFailed,
                .published_epoch = transaction.candidate->epoch,
                .migration_count = transaction.migrations.size(),
                .error = retirement_error.empty()
                             ? "Published ExpertOverlay epoch could not enter its retirement fence"
                             : std::move(retirement_error),
            };
        }

        return {
            .status = MoEOverlayResidencyApplyStatus::Published,
            .published_epoch = transaction.candidate->epoch,
            .migration_count = transaction.migrations.size(),
        };
    }

    MoEOverlayResidencyAuthorityStats
    MoEOverlayResidencyAuthority::stats() const noexcept
    {
        return {
            .checks = checks_.load(std::memory_order_relaxed),
            .capacity_bounded_proposals =
                capacity_bounded_proposals_.load(std::memory_order_relaxed),
            .bounded_candidate_snapshot_builds =
                bounded_candidate_snapshot_builds_.load(
                    std::memory_order_relaxed),
            .target_migrations_omitted =
                target_migrations_omitted_.load(std::memory_order_relaxed),
            .target_cycles_omitted =
                target_cycles_omitted_.load(std::memory_order_relaxed),
            .economy_proposals =
                economy_proposals_.load(std::memory_order_relaxed),
            .payoff_rejected_cycles =
                payoff_rejected_cycles_.load(std::memory_order_relaxed),
            .residency_rejected_cycles =
                residency_rejected_cycles_.load(std::memory_order_relaxed),
            .participant_rebalance_checks =
                participant_rebalance_checks_.load(
                    std::memory_order_relaxed),
            .participant_rebalance_proposals =
                participant_rebalance_proposals_.load(
                    std::memory_order_relaxed),
            .participant_rebalance_owner_changes =
                participant_rebalance_owner_changes_.load(
                    std::memory_order_relaxed),
            .static_no_movement_checks =
                static_no_movement_checks_.load(std::memory_order_relaxed),
            .dynamic_no_movement_checks =
                dynamic_no_movement_checks_.load(std::memory_order_relaxed),
            .committed_waves = committed_waves_.load(std::memory_order_relaxed),
            .committed_migrations =
                committed_migrations_.load(std::memory_order_relaxed),
            .committed_cycles =
                committed_cycles_.load(std::memory_order_relaxed),
            .prepared_context_restoration_waves =
                prepared_context_restoration_waves_.load(
                    std::memory_order_relaxed),
            .prepared_context_restoration_migrations =
                prepared_context_restoration_migrations_.load(
                    std::memory_order_relaxed),
            .prepared_context_restoration_cycles =
                prepared_context_restoration_cycles_.load(
                    std::memory_order_relaxed),
            .promotions = promotions_.load(std::memory_order_relaxed),
            .demotions = demotions_.load(std::memory_order_relaxed),
            .same_priority_moves =
                same_priority_moves_.load(std::memory_order_relaxed),
            .cross_domain_migrations =
                cross_domain_migrations_.load(std::memory_order_relaxed),
            .cross_rank_migrations =
                cross_rank_migrations_.load(std::memory_order_relaxed),
            .cross_backend_migrations =
                cross_backend_migrations_.load(std::memory_order_relaxed),
            .busy_rejections = busy_rejections_.load(std::memory_order_relaxed),
            .stale_rejections = stale_rejections_.load(std::memory_order_relaxed),
            .stage_failures = stage_failures_.load(std::memory_order_relaxed),
            .commit_failures = commit_failures_.load(std::memory_order_relaxed),
            .background_waves_started =
                background_waves_started_.load(std::memory_order_relaxed),
            .deferred_waves = deferred_waves_.load(std::memory_order_relaxed),
            .publication_boundary_reservations =
                publication_boundary_reservations_.load(
                    std::memory_order_relaxed),
            .publication_boundary_drains =
                publication_boundary_drains_.load(std::memory_order_relaxed),
            .graph_sequence_boundary_waits =
                graph_sequence_boundary_waits_.load(
                    std::memory_order_relaxed),
            .published_with_old_tickets =
                published_with_old_tickets_.load(std::memory_order_relaxed),
            .old_epoch_retirements =
                old_epoch_retirements_.load(std::memory_order_relaxed),
            .aborted_waves_reaped =
                aborted_waves_reaped_.load(std::memory_order_relaxed),
            .ticket_acquire_retries =
                ticket_acquire_retries_.load(std::memory_order_relaxed),
            .current_batch_llep_leases_acquired =
                current_batch_llep_leases_acquired_.load(
                    std::memory_order_relaxed),
            .current_batch_llep_leases_released =
                current_batch_llep_leases_released_.load(
                    std::memory_order_relaxed),
            .current_batch_llep_lease_rejections =
                current_batch_llep_lease_rejections_.load(
                    std::memory_order_relaxed),
        };
    }

    MoEOptimizationMovementLedger
    MoEOverlayResidencyAuthority::movementLedger() const
    {
        std::lock_guard<std::mutex> lock(movement_ledger_mutex_);
        return {
            .edges = movement_ledger_,
            .discarded_edges = 0u,
            .economy = movement_economy_,
            .discarded_economy_records = 0u,
            .host_admissions = movement_host_admissions_,
            .discarded_host_admission_records = 0u,
        };
    }

    size_t MoEOverlayResidencyAuthority::pendingRetirementCount() const noexcept
    {
        std::lock_guard<std::mutex> lock(maintenance_mutex_);
        return pending_retirements_.size();
    }

    size_t MoEOverlayResidencyAuthority::pendingAbortCount() const noexcept
    {
        std::lock_guard<std::mutex> lock(maintenance_mutex_);
        return pending_aborts_.size();
    }

    bool MoEOverlayResidencyAuthority::hasActiveBackgroundWave() const noexcept
    {
        std::lock_guard<std::mutex> lock(maintenance_mutex_);
        return active_wave_ != nullptr;
    }

    std::shared_ptr<const MoEOverlayResidencySnapshot>
    MoEOverlayResidencyAuthority::buildCalibrationShadowSnapshot(
        const std::shared_ptr<const MoEOverlayResidencySnapshot> &live)
    {
        if (!live || !live->valid() || live->epoch == std::numeric_limits<uint64_t>::max())
            throw std::invalid_argument("Calibration shadow requires a valid non-exhausted original source epoch");
        auto shadow = std::make_shared<MoEOverlayResidencySnapshot>(*live);
        shadow->epoch = live->epoch + 1;
        if (shadow->joint_)
            shadow->joint_->epoch = shadow->epoch;
        // All source/placement/control blocks remain identical. This private
        // epoch names shadow bytes only; no global-current selector is authored.
        if (!shadow->valid())
            throw std::logic_error("Calibration shadow lost its original sealed ownership arm");
        return shadow;
    }

    std::shared_ptr<const MoEOverlayResidencySnapshot>
    MoEOverlayResidencyAuthority::prepareCPUJointPlacementCandidate(
        const std::shared_ptr<const MoEOverlayResidencySnapshot> &expected_live,
        uint64_t expected_epoch, std::vector<RoutedExpertLayerPlacement> placements,
        std::span<const MoELayeredExpertOwnershipChange> whole_owner_changes) const
    {
        const auto previous = snapshot();
        const auto same_owner = [](const auto &a, const auto &b)
        { return a.get() == b.get() && !a.owner_before(b) && !b.owner_before(a); };
        if (!previous || !expected_live || !same_owner(previous, expected_live) ||
            previous->epoch != expected_epoch || !previous->valid() || !previous->joint_ ||
            !initial_snapshot_ || !initial_snapshot_->joint_ ||
            !same_owner(previous->joint_->source_plan, initial_snapshot_->joint_->source_plan) ||
            !same_owner(previous->joint_->preparation, config_.projection_preparation))
            throw std::logic_error("Joint candidate requires this authority's exact live epoch and "
                                   "original source publication");
        if (previous->epoch == std::numeric_limits<uint64_t>::max())
            throw std::overflow_error("Joint candidate epoch cannot advance past UINT64_MAX");
        // Only tier assignments are proposed. Original topology, precision,
        // cohort cuts, placement policy and preparation remain one authority.
        auto plan = *previous->joint_->source_plan;
        plan.placements = std::move(placements);
        const auto live =
            std::make_shared<const MoEExpertOwnerMap>(MoEExpertOwnerMap::buildShardTransition(
                plan, previous->placementOwnerMap(), previous->originalOwnerMap(), whole_owner_changes));
        auto candidate = std::make_shared<MoEOverlayResidencySnapshot>();
        candidate->epoch = previous->epoch + 1;
        candidate->placement_plan =
            std::make_shared<const MoERoutedExpertPlacementPlan>(std::move(plan));
        candidate->joint_.emplace(MoEOverlayResidencySnapshot::CPUJointDefinition{
            previous->joint_->source_plan, previous->joint_->preparation, candidate->epoch,
            candidate->placement_plan, live});
        if (!candidate->valid())
            throw std::logic_error(
                "Joint candidate producer did not seal a complete live placement");
        return candidate;
    }

    MoEOverlayResidencyTransaction
    MoEOverlayResidencyAuthority::prepareCPUJointPlacementTransaction(
        const std::shared_ptr<const MoEOverlayResidencySnapshot> &expected_live,
        uint64_t expected_epoch, std::vector<RoutedExpertLayerPlacement> placements,
        std::span<const MoELayeredExpertOwnershipChange> whole_owner_changes) const
    {
        MoEOverlayResidencyTransaction result;
        result.expected_epoch = expected_epoch;
        result.previous = expected_live;
        result.candidate = prepareCPUJointPlacementCandidate(
            expected_live, expected_epoch, std::move(placements), whole_owner_changes);
        requireCapacityPreserving(*result.previous, *result.candidate);
        result.migrations = buildMigrations(*result.previous, *result.candidate, nullptr, 0, whole_owner_changes);
        result.migration_cycles = buildMoEOverlayMigrationCycles(result.migrations);
        result.shadow_requirements = buildMoEOverlayShadowRequirements(result.migrations);
        if (!result.valid())
            throw std::logic_error("Joint placement producer did not author a complete transaction");
        return result;
    }

    std::shared_ptr<const MoEOverlayResidencySnapshot> MoEOverlayResidencyAuthority::buildSnapshot(
        uint64_t epoch, MoERoutedExpertPlacementPlan plan,
        const MoERoutedExpertModelMetadata &metadata, const MoEExpertOwnerMap *previous_owners,
        const MoELayeredExpertOwnership *explicit_ownership,
        std::shared_ptr<const MoEExpertOverlayPreparationPlan> joint_preparation)
    {
        if (epoch == 0)
            throw std::invalid_argument("MoE overlay residency epoch must be positive");
        if (plan.first_model_layer != metadata.first_model_layer)
            throw std::invalid_argument("MoE overlay snapshot plan belongs to another pipeline stage");
        const MoERoutedExpertPlacementValidationOptions validation_options{
            .layer_count = metadata.num_layers,
            .routed_expert_count = metadata.num_experts,
        };
        const auto validation = validateMoERoutedExpertPlacementPlan(
            plan,
            validation_options);
        if (!validation.ok())
        {
            std::ostringstream message;
            message << "Invalid MoE overlay residency snapshot:";
            for (const auto &error : validation.errors)
                message << "\n - " << error;
            throw std::invalid_argument(message.str());
        }

        if(jointCPUPlan(plan)) {
            if(epoch!=1 || previous_owners || explicit_ownership || !joint_preparation)
                throw std::invalid_argument("CPU joint snapshot requires its original Static source preparation, never scalar migration");
            requireJointSource(plan,metadata,*joint_preparation);
            auto snapshot=std::make_shared<MoEOverlayResidencySnapshot>();
            snapshot->epoch=epoch;snapshot->placement_plan=std::make_shared<const MoERoutedExpertPlacementPlan>(std::move(plan));
            const auto original_owners = joint_preparation->cpuJointOwnerMap();
            snapshot->joint_.emplace(MoEOverlayResidencySnapshot::CPUJointDefinition{
                snapshot->placement_plan, std::move(joint_preparation), epoch,
                snapshot->placement_plan, original_owners});
            return snapshot;
        }
        if(joint_preparation && joint_preparation->cpuJointOwnerMap())
            throw std::invalid_argument("Joint source preparation cannot be rebound to a whole-expert plan");
        auto mutable_plan = std::make_shared<MoERoutedExpertPlacementPlan>(
            std::move(plan));
        /*
         * Epoch transitions preserve retained participant owners. Rebuilding
         * contiguous chunks from scratch would turn one logical tier swap into
         * unrelated same-tier copies, changing both the transfer BOM and the
         * measured economy identity of a bounded wave.
         */
        MoEExpertOwnerMap owner_map = explicit_ownership
                                          ? MoEExpertOwnerMap::buildExplicit(
                                                *mutable_plan,
                                                *explicit_ownership)
                                          : (previous_owners
                                                 ? MoEExpertOwnerMap::buildTransition(
                                                       *mutable_plan,
                                                       *previous_owners)
                                                 : MoEExpertOwnerMap::build(
                                                       *mutable_plan));
        auto layered = owner_map.layeredOwnership(
            metadata.num_layers,
            metadata.num_experts, metadata.first_model_layer);

        auto snapshot = std::make_shared<MoEOverlayResidencySnapshot>();
        snapshot->epoch = epoch;
        snapshot->placement_plan = std::move(mutable_plan);
        snapshot->owner_map = std::move(owner_map);
        snapshot->layered_ownership = std::move(layered);
        return snapshot;
    }

    std::vector<MoEOverlayTierMigration>
    MoEOverlayResidencyAuthority::buildMigrations(
        const MoEOverlayResidencySnapshot &previous,
        const MoEOverlayResidencySnapshot &candidate,
        const ValidatedDecodeExpertHistogramWindowView *histogram_window,
        size_t estimated_weight_bytes,
        std::span<const MoELayeredExpertOwnershipChange>
            participant_changes)
    {
        using ExpertKey = std::pair<int, int>;
        std::map<ExpertKey, int> participant_destinations;
        for (const auto &change : participant_changes)
        {
            // Objectives may belong to a closed cycle excluded from this
            // bounded subset. Only an actual matching destination below claims
            // that axis; cohort rows never acquire a scalar owner from this list.
            if (change.layer_idx < 0 || change.expert_id < 0 ||
                change.previous_participant < 0 ||
                change.current_participant < 0 ||
                change.previous_participant == change.current_participant ||
                change.layer_idx < candidate.firstModelLayer() ||
                change.layer_idx - candidate.firstModelLayer() >= candidate.layerCount() ||
                change.expert_id >= candidate.expertCount() ||
                !candidate.placementOwnerMap().participantForId(change.current_participant) ||
                !participant_destinations
                     .emplace(
                         ExpertKey{change.layer_idx, change.expert_id},
                         change.current_participant)
                     .second)
            {
                throw std::logic_error(
                    "ExpertOverlay participant objective does not name one unique candidate owner change");
            }
        }

        std::vector<MoEOverlayTierMigration> migrations;
        for (const auto &destination : candidate.placementOwnerMap().owners())
        {
            const auto *source = previous.placementOwnerMap().placementFor(
                destination.layer_idx,
                destination.expert_id);
            if (!source)
            {
                throw std::logic_error(
                    "Candidate MoE overlay owner has no source owner");
            }
            if (*source == destination)
            {
                continue;
            }

            const int source_priority = tierPriority(
                *previous.placement_plan,
                source->tier_idx);
            const int destination_priority = tierPriority(
                *candidate.placement_plan,
                destination.tier_idx);
            const auto direction = destination_priority < source_priority
                                       ? MoEOverlayTierMigrationDirection::Promotion
                                       : (destination_priority > source_priority
                                              ? MoEOverlayTierMigrationDirection::Demotion
                                              : MoEOverlayTierMigrationDirection::SamePriority);
            const auto participant_destination =
                participant_destinations.find({
                    destination.layer_idx,
                    destination.expert_id,
                });
            const bool advances_participant =
                participant_destination != participant_destinations.end() &&
                participant_destination->second ==
                    destination.owner_participant;
            const auto axis = advances_participant
                                  ? (source->tier_idx != destination.tier_idx
                                         ? MoEOptimizationMovementAxis::Combined
                                         : MoEOptimizationMovementAxis::ParticipantPlacement)
                                  : MoEOptimizationMovementAxis::TierResidency;
            migrations.push_back({
                .layer_idx = destination.layer_idx,
                .expert_id = destination.expert_id,
                .activation_count = histogram_window
                                        ? histogram_window->activationCount(
                                              destination.layer_idx,
                                              destination.expert_id)
                                        : 0,
                .estimated_weight_bytes = estimated_weight_bytes,
                .direction = direction,
                .axis = axis,
                .source = *source,
                .destination = destination,
            });
        }

        std::sort(
            migrations.begin(),
            migrations.end(),
            [](const auto &lhs, const auto &rhs)
            {
                if (lhs.layer_idx != rhs.layer_idx)
                    return lhs.layer_idx < rhs.layer_idx;
                return lhs.expert_id < rhs.expert_id;
            });
        return migrations;
    }

    std::vector<MoEOverlayTierMigrationCycle> decomposeMoEOverlayMigrationCycles(
        const std::vector<MoEOverlayTierMigration> &migrations)
    {
        if (!analyzeMoEOverlayMigrationCapacity(migrations).capacityPreserved())
            throw std::logic_error("MoE migration requires complete capacity-preserving placement targets");
        // Buckets only narrow equality checks; the full original typed target
        // (including cohort source and cuts) is the vertex identity. No pointer
        // identity, target hash, physical owner ledger or sorted edge order is
        // introduced. Distinct valid cuts may share all bucket scalars.
        struct Vertex
        {
            MoEExpertPlacementTarget target;
            std::vector<size_t> outgoing;
            size_t active_epoch = 0;
        };
        using VertexBucket = std::tuple<int, int, bool, int>;
        std::map<VertexBucket, std::vector<size_t>> buckets;
        std::vector<Vertex> vertices;
        vertices.reserve(migrations.size());
        const auto vertex_for = [&](int layer, MoEExpertPlacementTarget target)
        {
            auto &bucket = buckets[{layer, target.tierIndex(), target.isCohort(), target.participantId(0)}];
            for (const size_t vertex : bucket)
                if (vertices[vertex].target == target)
                    return vertex;
            const size_t vertex = vertices.size();
            vertices.push_back({std::move(target), {}, 0});
            bucket.push_back(vertex);
            return vertex;
        };
        std::vector<std::pair<size_t, size_t>> targets;
        targets.reserve(migrations.size());
        for (size_t index = 0; index < migrations.size(); ++index)
        {
            const auto &edge = migrations[index];
            const auto original = *migrationPlacementTargets(edge);
            const size_t source = vertex_for(edge.layer_idx, original.first);
            const size_t destination = vertex_for(edge.layer_idx, original.second);
            targets.emplace_back(source, destination);
            // Original input order is the DFS tie authority, even when the
            // caller supplies unsorted layers or interleaved physical targets.
            vertices[source].outgoing.push_back(index);
        }
        std::vector<MoEOverlayTierMigrationCycle> cycles;
        std::vector<bool> consumed(migrations.size(), false);
        std::vector<size_t> return_path;
        for (size_t seed_index = 0; seed_index < migrations.size(); ++seed_index)
        {
            if (consumed[seed_index])
                continue;
            const auto &seed = migrations[seed_index];
            consumed[seed_index] = true;
            MoEOverlayTierMigrationCycle cycle{.layer_idx = seed.layer_idx,
                .migration_indices = {seed_index}};
            if (targets[seed_index].first != targets[seed_index].second)
            {
                return_path.clear();
                // seed_index < migrations.size(), so this positive traversal
                // epoch cannot overflow size_t. Failed branches clear their
                // active mark; successful-path marks expire at the next seed.
                const size_t search_epoch = seed_index + 1u;
                const auto find_return_path = [&](auto &&self, size_t current) -> bool
                {
                    if (current == targets[seed_index].first)
                        return true;
                    auto &vertex = vertices[current];
                    if (vertex.active_epoch == search_epoch)
                        return false;
                    vertex.active_epoch = search_epoch;
                    // Exact vertex adjacency is the original global scan with
                    // only its always-rejected layer/source rows removed.
                    for (const size_t index : vertex.outgoing)
                    {
                        if (consumed[index])
                            continue;
                        consumed[index] = true;
                        return_path.push_back(index);
                        if (self(self, targets[index].second))
                            return true;
                        return_path.pop_back();
                        consumed[index] = false;
                    }
                    vertex.active_epoch = 0;
                    return false;
                };
                if (!find_return_path(find_return_path, targets[seed_index].second))
                    throw std::logic_error("MoE placement delta cannot form complete whole/cohort cycles");
                cycle.migration_indices.insert(cycle.migration_indices.end(),
                    return_path.begin(), return_path.end());
            }
            if (!cycle.valid(migrations))
                throw std::logic_error("MoE overlay produced an invalid placement migration cycle");
            cycles.push_back(std::move(cycle));
        }
        return cycles;
    }

    std::vector<MoEOverlayTierMigrationCycle> buildMoEOverlayMigrationCycles(
        std::vector<MoEOverlayTierMigration> &migrations)
    {
        // Physical closure may combine independent logical objectives. Keep
        // each command's authored axis; migrationCycleAxis computes the union
        // only where a scheduler or economy decision needs the whole cycle.
        return decomposeMoEOverlayMigrationCycles(migrations);
    }

    std::vector<MoEOverlayTierShadowRequirement> buildMoEOverlayShadowRequirements(
        const std::vector<MoEOverlayTierMigration> &migrations)
    {
        using RequirementKey = std::tuple<int, int, int>;
        std::map<RequirementKey, size_t> counts;
        std::set<std::pair<int, int>> logical_edges;
        for (const auto &migration : migrations)
        {
            const auto targets = migrationPlacementTargets(migration);
            if (!targets || !logical_edges.emplace(migration.layer_idx, migration.expert_id).second)
                throw std::logic_error("MoE shadow slots require complete original placement targets");
            const auto &destination = targets->second;
            for (size_t member = 0; member < destination.participantCount(); ++member)
                ++counts[{migration.layer_idx, destination.tierIndex(), destination.participantId(member)}];
        }
        std::vector<MoEOverlayTierShadowRequirement> requirements;
        requirements.reserve(counts.size());
        for (const auto &[key, count] : counts)
        {
            const auto &[layer, tier, participant] = key;
            requirements.push_back({.layer_idx = layer, .tier_idx = tier,
                .destination_participant = participant, .slot_count = count});
        }
        return requirements;
    }

    void MoEOverlayResidencyAuthority::requireCapacityPreserving(
        const MoEOverlayResidencySnapshot &previous,
        const MoEOverlayResidencySnapshot &candidate)
    {
        if (tierCapacities(*previous.placement_plan) !=
            tierCapacities(*candidate.placement_plan))
        {
            throw std::logic_error(
                "MoE overlay rebalance changed fixed per-layer tier capacity");
        }
        if (participantCapacities(previous.placementOwnerMap()) !=
            participantCapacities(candidate.placementOwnerMap()))
        {
            throw std::logic_error(
                "MoE overlay rebalance changed fixed per-layer participant capacity");
        }
    }

    void MoEOverlayResidencyAuthority::releaseTicket(
        const std::shared_ptr<PublishedEpochState> &epoch_state,
        TicketLeasePurpose purpose) noexcept
    {
        if (!epoch_state)
            std::terminate();

        const uint64_t previous_epoch_count =
            epoch_state->active_tickets.fetch_sub(1, std::memory_order_acq_rel);
        const uint64_t previous_total_count =
            active_ticket_count_.fetch_sub(1, std::memory_order_acq_rel);
        if (previous_epoch_count == 0 || previous_total_count == 0)
            std::terminate();
        if (purpose == TicketLeasePurpose::InferenceGraphSequence)
        {
            const uint64_t previous_graph_sequences =
                epoch_state->active_graph_sequences.fetch_sub(
                    1, std::memory_order_acq_rel);
            if (previous_graph_sequences == 0)
                std::terminate();
        }
        if (purpose == TicketLeasePurpose::CurrentBatchLLEP)
        {
            current_batch_llep_leases_released_.fetch_add(
                1, std::memory_order_relaxed);
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "current_batch_llep_leases_released",
                1.0,
                "prefill",
                config_.perf_device,
                {{"epoch",
                  std::to_string(epoch_state->snapshot
                                     ? epoch_state->snapshot->epoch
                                     : 0)}});
        }
        else if (purpose == TicketLeasePurpose::InferenceGraphSequence)
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "graph_sequence_epoch_leases",
                1.0,
                "inference",
                config_.perf_device,
                {{"action", "release"},
                 {"epoch",
                  std::to_string(epoch_state->snapshot
                                     ? epoch_state->snapshot->epoch
                                     : 0)}});
        }
    }

    bool MoEOverlayResidencyAuthority::reapReadyRetirementsLocked(
        std::string *error)
    {
        if (error)
            error->clear();
        auto retirement = pending_retirements_.begin();
        while (retirement != pending_retirements_.end())
        {
            auto &pending = **retirement;

            const bool admission_open =
                pending.local_grace_period ==
                PendingRetirement::LocalGracePeriodState::AdmissionOpen;
            const bool local_readers_drained =
                pending.previous->active_tickets.load(
                    std::memory_order_acquire) == 0;
            const MoEOverlayLocalRetirementState local_state{
                .admission =
                    admission_open
                        ? MoEOverlayRetirementAdmissionState::Open
                        : MoEOverlayRetirementAdmissionState::Closed,
                .readers =
                    local_readers_drained
                        ? MoEOverlayRetirementReaderState::Drained
                        : MoEOverlayRetirementReaderState::Active,
            };

            std::string fence_error;
            const auto fence_progress =
                pending.work->pollRetirementFence(
                    local_state, &fence_error);
            if (fence_progress == MoEOverlayRetirementFenceProgress::Pending)
            {
                ++retirement;
                continue;
            }
            if (fence_progress == MoEOverlayRetirementFenceProgress::Failed)
            {
                if (error)
                {
                    *error = fence_error.empty()
                                 ? "ExpertOverlay old-epoch retirement fence failed"
                                 : std::move(fence_error);
                }
                return false;
            }

            if (fence_progress ==
                MoEOverlayRetirementFenceProgress::ReadyToCloseAdmission)
            {
                if (!admission_open || !local_readers_drained)
                {
                    if (error)
                    {
                        *error =
                            "ExpertOverlay retirement-admission fence became ready from an invalid local grace-period state";
                    }
                    return false;
                }

                /* Closing admission is the only inference-visible edge in
                 * retirement. An acquisition racing this store either owns a
                 * counted reader or drops its provisional pin after observing
                 * the closed gate; the phase-6 drain covers both outcomes. */
                pending.previous->accepting_exact_tickets.store(
                    false, std::memory_order_release);
                pending.local_grace_period =
                    PendingRetirement::LocalGracePeriodState::AdmissionClosed;
                continue;
            }

            if (fence_progress !=
                MoEOverlayRetirementFenceProgress::ReadyToRetire)
            {
                if (error)
                    *error = "ExpertOverlay retirement fence returned an unknown state";
                return false;
            }

            /* Closed admission makes zero readers stable. Re-read the source
             * counter before reclaiming so a racing pre-close acquisition must
             * have completed its final release. */
            if (admission_open ||
                pending.previous->active_tickets.load(
                    std::memory_order_acquire) != 0)
            {
                if (error)
                {
                    *error =
                        "ExpertOverlay retirement fence became ready before exact admission closed and readers drained";
                }
                return false;
            }

            auto retiring_state = pending.previous;
            if (!retiring_epoch_.compare_exchange_strong(
                    retiring_state,
                    std::shared_ptr<PublishedEpochState>{},
                    std::memory_order_release,
                    std::memory_order_acquire))
            {
                if (error)
                {
                    *error =
                        "ExpertOverlay exact-ticket retirement slot does not name the retiring epoch";
                }
                return false;
            }

            /*
             * Final ticket return only decrements an atomic. Reclamation is
             * deliberately performed here on the maintenance worker, after
             * the wave's optional all-rank fence, so an inference/collective
             * return thread never destroys engines or releases transport
             * objects and no peer can still be sending the retired epoch.
             */
            pending.work->retirePrevious();
            old_epoch_retirements_.fetch_add(1, std::memory_order_relaxed);
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "old_epoch_retirements",
                1.0,
                "maintenance",
                config_.perf_device,
                baseTags(
                    config_.initial_plan.residency_policy,
                    pending.previous->snapshot->epoch));
            retirement = pending_retirements_.erase(retirement);
        }
        return true;
    }

    void MoEOverlayResidencyAuthority::reapReadyAbortsLocked() noexcept
    {
        auto pending_abort = pending_aborts_.begin();
        while (pending_abort != pending_aborts_.end())
        {
            std::string cleanup_error;
            const auto progress =
                (*pending_abort)->work->pollAbort(&cleanup_error);
            if (progress == MoEOverlayResidencyWaveProgress::Pending)
            {
                ++pending_abort;
                continue;
            }
            if (progress == MoEOverlayResidencyWaveProgress::Failed)
            {
                /*
                 * Losing the completion authority makes safe buffer/event
                 * reclamation unknowable. Continuing would either leak a live
                 * bank indefinitely or risk use-after-free, so this is fatal.
                 */
                LOG_ERROR(
                    "MoE overlay abort cleanup failed for epoch "
                    << (*pending_abort)->previous_epoch << ": "
                    << (cleanup_error.empty()
                            ? "background cleanup returned no diagnostic"
                            : cleanup_error));
                std::terminate();
            }

            recordAbortedWaveReapedLocked(
                (*pending_abort)->previous_epoch);
            pending_abort = pending_aborts_.erase(pending_abort);
        }
    }

    void MoEOverlayResidencyAuthority::recordAbortedWaveReapedLocked(
        uint64_t epoch) noexcept
    {
        aborted_waves_reaped_.fetch_add(1, std::memory_order_relaxed);
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "aborted_waves_reaped",
            1.0,
            "maintenance",
            config_.perf_device,
            baseTags(config_.initial_plan.residency_policy, epoch));
    }

    void MoEOverlayResidencyAuthority::abortAndRetainWaveLocked(
        std::unique_ptr<IMoEOverlayResidencyWave> work,
        uint64_t epoch) noexcept
    {
        if (!work)
            return;

        work->abortStaged();
        std::string cleanup_error;
        const auto cleanup_progress = work->pollAbort(&cleanup_error);
        if (cleanup_progress == MoEOverlayResidencyWaveProgress::Pending)
        {
            auto pending = std::make_unique<PendingAbort>();
            pending->previous_epoch = epoch;
            pending->work = std::move(work);
            pending_aborts_.push_back(std::move(pending));
            return;
        }
        if (cleanup_progress == MoEOverlayResidencyWaveProgress::Ready)
        {
            recordAbortedWaveReapedLocked(epoch);
            return;
        }

        LOG_ERROR(
            "MoE overlay abort cleanup failed for epoch " << epoch << ": "
            << (cleanup_error.empty()
                    ? "background cleanup returned no diagnostic"
                    : cleanup_error));
        std::terminate();
    }

    MoEOverlayResidencyApplyResult
    MoEOverlayResidencyAuthority::failActiveWaveLocked(
        MoEOverlayResidencyApplyStatus status,
        std::string error) noexcept
    {
        const uint64_t epoch =
            active_wave_ && active_wave_->previous &&
                    active_wave_->previous->snapshot
                ? active_wave_->previous->snapshot->epoch
                : 0;

        if (active_wave_ && active_wave_->previous &&
            active_wave_->phase == ActiveBackgroundWave::Phase::
                                       AwaitingGraphSequenceBoundary)
        {
            /* No selector has changed yet, so aborting a prepared wave safely
             * reopens ordinary sequence admission on the still-current epoch. */
            active_wave_->previous->graph_sequence_admission.store(
                PublishedEpochState::GraphSequenceAdmissionState::Open,
                std::memory_order_release);
            active_wave_->previous->graph_sequence_admission.notify_all();
        }

        if (active_wave_ && active_wave_->work)
        {
            abortAndRetainWaveLocked(
                std::move(active_wave_->work),
                epoch);
        }
        active_wave_.reset();

        const char *counter_name = nullptr;
        if (status == MoEOverlayResidencyApplyStatus::StageFailed)
        {
            stage_failures_.fetch_add(1, std::memory_order_relaxed);
            counter_name = "migration_stage_failures";
        }
        else if (status ==
                 MoEOverlayResidencyApplyStatus::PreparationFailed)
        {
            commit_failures_.fetch_add(1, std::memory_order_relaxed);
            counter_name = "migration_preparation_failures";
        }

        if (counter_name)
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                counter_name,
                1.0,
                "maintenance",
                config_.perf_device,
                baseTags(config_.initial_plan.residency_policy, epoch));
        }

        return {
            .status = status,
            .published_epoch = epoch,
            .error = std::move(error),
        };
    }

    void MoEOverlayResidencyAuthority::recordNoMovement(bool static_policy)
    {
        if (static_policy)
            static_no_movement_checks_.fetch_add(1, std::memory_order_relaxed);
        else
            dynamic_no_movement_checks_.fetch_add(1, std::memory_order_relaxed);

        const auto current = snapshot();
        auto tags = baseTags(
            config_.initial_plan.residency_policy,
            current ? current->epoch : 0);
        tags.emplace("movement", "none");
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            static_policy ? "static_no_movement_checks"
                          : "dynamic_no_movement_checks",
            1.0,
            "maintenance",
            config_.perf_device,
            tags);
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "committed_expert_migrations",
            0.0,
            "maintenance",
            config_.perf_device,
            tags);
    }

    void MoEOverlayResidencyAuthority::recordCommitted(
        const MoEOverlayResidencyTransaction &transaction)
    {
        const bool prepared_context_restoration =
            transaction.purpose ==
            MoEOverlayResidencyTransactionPurpose::PreparedContextRestoration;
        if (transaction.economy.enabled)
        {
            std::lock_guard<std::mutex> economy_lock(economy_mutex_);
            if (!economy_state_)
                std::terminate();
            for (const auto &migration : transaction.migrations)
            {
                economy_state_->last_moved_generation[
                    economy_state_->expertOffset(
                        migration.layer_idx,
                        migration.expert_id)] =
                    transaction.histogram_generation;
            }
        }

        uint64_t promotions = 0;
        uint64_t demotions = 0;
        uint64_t same_priority = 0;
        uint64_t cross_domain = 0;
        uint64_t cross_rank = 0;
        uint64_t cross_backend = 0;
        size_t estimated_bytes = 0;
        std::vector<MoEOptimizationMovementEdge> committed_edges;
        std::optional<MoEOptimizationMovementEconomy> committed_economy;
        std::optional<MoEOptimizationHostMovementAdmission>
            committed_host_admission;
        if (!prepared_context_restoration)
        {
            committed_edges.reserve(transaction.migrations.size());
            if (transaction.host_admission)
            {
                committed_host_admission = *transaction.host_admission;
                if (!committed_host_admission->valid() ||
                    committed_host_admission->transaction !=
                        transaction.candidate->epoch ||
                    committed_host_admission->admitted_physical_cycles !=
                        transaction.migration_cycles.size())
                {
                    /* Policy admission is operational evidence owned by the
                     * coordinator. Telemetry cannot repair it at commit. */
                    std::terminate();
                }
            }
            if (transaction.economy.enabled)
            {
                committed_economy = MoEOptimizationMovementEconomy{
                    .authority = MoEOptimizationAuthority::Host,
                    .transaction = transaction.candidate->epoch,
                    .candidate_epoch = transaction.candidate->epoch,
                    .command_count = static_cast<std::uint64_t>(
                        transaction.migrations.size()),
                    .cycle_count = static_cast<std::uint64_t>(
                        transaction.migration_cycles.size()),
                    .proof = MoEOptimizationTimeEconomy{
                    .projected_service_gain_ns =
                        transaction.economy.projected_service_gain_ns,
                    .projected_transfer_and_repack_ns =
                        transaction.economy
                            .projected_transfer_and_repack_ns,
                    .projected_inference_interference_ns =
                        transaction.economy
                            .projected_inference_interference_ns,
                    .projected_net_benefit_ns =
                        transaction.economy.projected_net_benefit_ns,
                    },
                };
                if (!committed_economy->valid())
                {
                    /* A committed Dynamic policy decision must retain the
                     * exact arithmetic that admitted it. Optional telemetry
                     * cannot repair a malformed authority-owned proof. */
                    std::terminate();
                }
            }
        }

        const auto capacity_evidence =
            analyzeMoEOverlayMigrationCapacity(transaction.migrations);
        if (!capacity_evidence.capacityPreserved())
        {
            // Publication must never turn a malformed slot-flow graph into a
            // live epoch. `valid()` rejects this earlier; keep the commit edge
            // independently fatal because it is the sole authority boundary.
            std::terminate();
        }

        std::vector<size_t> cycle_index_by_migration(
            transaction.migrations.size(),
            std::numeric_limits<size_t>::max());
        std::vector<size_t> cycle_size_by_migration(
            transaction.migrations.size(), 0u);
        for (size_t cycle_index = 0;
             cycle_index < transaction.migration_cycles.size();
             ++cycle_index)
        {
            const auto &cycle = transaction.migration_cycles[cycle_index];
            for (const size_t migration_index : cycle.migration_indices)
            {
                if (migration_index >= transaction.migrations.size() ||
                    cycle_index_by_migration[migration_index] !=
                        std::numeric_limits<size_t>::max())
                {
                    std::terminate();
                }
                cycle_index_by_migration[migration_index] = cycle_index;
                cycle_size_by_migration[migration_index] =
                    cycle.migration_indices.size();
            }
        }

        for (size_t migration_index = 0;
             migration_index < transaction.migrations.size();
             ++migration_index)
        {
            const auto &migration = transaction.migrations[migration_index];
            if (cycle_index_by_migration[migration_index] ==
                    std::numeric_limits<size_t>::max() ||
                cycle_size_by_migration[migration_index] == 0u)
            {
                std::terminate();
            }
            const int source_priority = tierPriority(
                *transaction.previous->placement_plan,
                migration.source.tier_idx);
            const int destination_priority = tierPriority(
                *transaction.candidate->placement_plan,
                migration.destination.tier_idx);
            switch (migration.direction)
            {
            case MoEOverlayTierMigrationDirection::Promotion:
                ++promotions;
                break;
            case MoEOverlayTierMigrationDirection::Demotion:
                ++demotions;
                break;
            case MoEOverlayTierMigrationDirection::SamePriority:
                ++same_priority;
                break;
            }
            cross_domain += migration.crossesDomain() ? 1u : 0u;
            cross_rank += migration.crossesWorldRank() ? 1u : 0u;
            cross_backend += migration.crossesBackend() ? 1u : 0u;
            estimated_bytes += migration.estimated_weight_bytes;

            if (!prepared_context_restoration)
            {
                MoEOptimizationMovementDirection typed_direction =
                    MoEOptimizationMovementDirection::SamePriority;
                if (migration.direction ==
                    MoEOverlayTierMigrationDirection::Promotion)
                {
                    typed_direction =
                        MoEOptimizationMovementDirection::Promotion;
                }
                else if (migration.direction ==
                         MoEOverlayTierMigrationDirection::Demotion)
                {
                    typed_direction =
                        MoEOptimizationMovementDirection::Demotion;
                }
                committed_edges.push_back({
                    .authority = MoEOptimizationAuthority::Host,
                    .transaction = transaction.candidate->epoch,
                    .candidate_epoch = transaction.candidate->epoch,
                    .layer = migration.layer_idx,
                    .expert = migration.expert_id,
                    .cycle_index =
                        cycle_index_by_migration[migration_index],
                    .cycle_size =
                        cycle_size_by_migration[migration_index],
                    .direction = typed_direction,
                    .axis = migration.axis,
                    .source = MoEOptimizationMovementEndpoint::fromOwner(migration.source),
                    .destination = MoEOptimizationMovementEndpoint::fromOwner(migration.destination),
                    .source_priority = source_priority,
                    .destination_priority = destination_priority,
                    .estimated_weight_bytes =
                        static_cast<std::uint64_t>(
                            migration.estimated_weight_bytes),
                    .activation_count = migration.activation_count,
                    .blocking_inference = false,
                });
            }

            // A committed wave may contain hundreds of edges on every rank.
            // Keep individual edges available for deep diagnosis without
            // making formatting and terminal I/O part of ordinary inference.
            // The typed committed-edge evidence above and the wave-level INFO
            // summary remain the production authorities.
            LOG_TRACE(
                "[ExpertOverlay][Residency] committed edge epoch="
                << transaction.candidate->epoch
                << " purpose="
                << (prepared_context_restoration
                        ? "prepared_context_restoration"
                        : "live_placement_change")
                << " layer=" << migration.layer_idx
                << " expert=" << migration.expert_id
                << " direction=" << directionName(migration.direction)
                << " movement_axis=" << movementAxisName(migration.axis)
                << " source_priority=" << source_priority
                << " destination_priority=" << destination_priority
                << " source_target="
                << economyExecutionTarget(migration.source).word()
                << " source_rank="
                << (migration.source.owner_world_rank_known
                        ? std::to_string(
                              migration.source.owner_world_rank)
                        : "unknown")
                << " destination_target="
                << economyExecutionTarget(migration.destination).word()
                << " destination_rank="
                << (migration.destination.owner_world_rank_known
                        ? std::to_string(
                              migration.destination.owner_world_rank)
                        : "unknown")
                << " crosses_rank="
                << (migration.crossesWorldRank() ? "true" : "false"));

            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                prepared_context_restoration
                    ? "prepared_context_restoration_edges"
                    : "expert_migration_edges",
                1.0,
                prepared_context_restoration ? "model_teardown"
                                             : "maintenance",
                config_.perf_device,
                {
                    {"epoch", std::to_string(transaction.candidate->epoch)},
                    {"candidate_epoch",
                     std::to_string(transaction.candidate->epoch)},
                    {"cycle_index",
                     std::to_string(
                         cycle_index_by_migration[migration_index])},
                    {"cycle_size",
                     std::to_string(
                         cycle_size_by_migration[migration_index])},
                    {"layer", std::to_string(migration.layer_idx)},
                    {"expert", std::to_string(migration.expert_id)},
                    {"direction", directionName(migration.direction)},
                    {"movement_axis", movementAxisName(migration.axis)},
                    {"source_tier", migration.source.tier_name},
                    {"destination_tier", migration.destination.tier_name},
                    {"source_priority", std::to_string(source_priority)},
                    {"destination_priority",
                     std::to_string(destination_priority)},
                    {"source_domain", migration.source.domain_name},
                    {"destination_domain", migration.destination.domain_name},
                    {migration.source.shard_cohort ? "source_target" : "source_participant",
                     std::to_string(economyExecutionTarget(migration.source).word())},
                    {migration.destination.shard_cohort ? "destination_target" : "destination_participant",
                     std::to_string(economyExecutionTarget(migration.destination).word())},
                    {"source_world_rank",
                     migration.source.owner_world_rank_known
                         ? std::to_string(
                               migration.source.owner_world_rank)
                         : "unknown"},
                    {"destination_world_rank",
                     migration.destination.owner_world_rank_known
                         ? std::to_string(
                               migration.destination.owner_world_rank)
                         : "unknown"},
                    {"crosses_world_rank",
                     migration.crossesWorldRank() ? "true" : "false"},
                    {"source_device", migration.source.shard_cohort ? "cpu_cohort" : migration.source.device.toString()},
                    {"destination_device", migration.destination.shard_cohort ? "cpu_cohort" : migration.destination.device.toString()},
                    {"estimated_weight_bytes",
                     std::to_string(migration.estimated_weight_bytes)},
                    {"activation_count", std::to_string(migration.activation_count)},
                    {"blocking_inference", "false"},
                    {"policy_owner", "host"},
                    {"transaction_purpose",
                     prepared_context_restoration
                         ? "prepared_context_restoration"
                         : "live_placement_change"},
                });
        }

        if (!committed_edges.empty() || committed_economy ||
            committed_host_admission)
        {
            /* Publish edge identities and coordinator-owned economics in one
             * critical section. Readers can never observe a movement without
             * the admitting proof on its policy-authority rank. */
            std::lock_guard<std::mutex> ledger_lock(
                movement_ledger_mutex_);
            movement_ledger_.insert(
                movement_ledger_.end(),
                committed_edges.begin(),
                committed_edges.end());
            if (committed_economy)
                movement_economy_.push_back(*committed_economy);
            if (committed_host_admission)
            {
                movement_host_admissions_.push_back(
                    *committed_host_admission);
            }
        }

        if (prepared_context_restoration)
        {
            prepared_context_restoration_waves_.fetch_add(
                1, std::memory_order_relaxed);
            prepared_context_restoration_migrations_.fetch_add(
                transaction.migrations.size(), std::memory_order_relaxed);
            prepared_context_restoration_cycles_.fetch_add(
                transaction.migration_cycles.size(),
                std::memory_order_relaxed);
        }
        else
        {
            if (PerfStatsCollector::isDomainEnabled("moe_overlay_residency"))
            {
                /* The transport folds the same immutable identity only after
                 * publishing the fully prepared runtime bank. This independent
                 * owner witness follows ledger publication, so diagnostics can
                 * match their complete ordering without per-epoch telemetry keys.
                 * Neither witness participates in the runtime commit decision. */
                PerfStatsCollector::recordOrderedSequenceStep(
                    "moe_overlay_residency",
                    "placement_owner_publications",
                    {transaction.candidate->epoch,
                     static_cast<std::uint64_t>(transaction.migrations.size())},
                    "maintenance", config_.perf_device,
                    {{"purpose", "placement_change"}});
            }
            committed_waves_.fetch_add(1, std::memory_order_relaxed);
            committed_migrations_.fetch_add(
                transaction.migrations.size(),
                std::memory_order_relaxed);
            committed_cycles_.fetch_add(
                transaction.migration_cycles.size(),
                std::memory_order_relaxed);
            promotions_.fetch_add(promotions, std::memory_order_relaxed);
            demotions_.fetch_add(demotions, std::memory_order_relaxed);
            same_priority_moves_.fetch_add(
                same_priority, std::memory_order_relaxed);
            cross_domain_migrations_.fetch_add(
                cross_domain, std::memory_order_relaxed);
            cross_rank_migrations_.fetch_add(
                cross_rank, std::memory_order_relaxed);
            cross_backend_migrations_.fetch_add(
                cross_backend, std::memory_order_relaxed);
        }

        auto tags = baseTags(
            config_.initial_plan.residency_policy,
            transaction.candidate->epoch);
        tags.emplace(
            "transaction_purpose",
            prepared_context_restoration
                ? "prepared_context_restoration"
                : "live_placement_change");
        const auto add = [&](const char *name, double value)
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                name,
                value,
                prepared_context_restoration ? "model_teardown"
                                             : "maintenance",
                config_.perf_device,
                tags);
        };
        if (prepared_context_restoration)
        {
            add("prepared_context_restoration_waves", 1.0);
            add("prepared_context_restoration_migrations",
                static_cast<double>(transaction.migrations.size()));
            add("prepared_context_restoration_cycles",
                static_cast<double>(transaction.migration_cycles.size()));
            add("prepared_context_restoration_estimated_weight_bytes",
                static_cast<double>(estimated_bytes));
        }
        else
        {
            add("committed_waves", 1.0);
            add("committed_expert_migrations",
                static_cast<double>(transaction.migrations.size()));
            add("committed_migration_cycles",
                static_cast<double>(transaction.migration_cycles.size()));
            add("promotions", static_cast<double>(promotions));
            add("demotions", static_cast<double>(demotions));
            add("same_priority_moves", static_cast<double>(same_priority));
            add("cross_domain_migrations", static_cast<double>(cross_domain));
            add("cross_rank_migrations", static_cast<double>(cross_rank));
            add("cross_backend_migrations", static_cast<double>(cross_backend));
            add("estimated_weight_bytes", static_cast<double>(estimated_bytes));
        }
        auto capacity_tags = tags;
        capacity_tags.emplace(
            "edges_checked",
            std::to_string(capacity_evidence.edges_checked));
        capacity_tags.emplace(
            "closed_cycles",
            std::to_string(transaction.migration_cycles.size()));
        capacity_tags.emplace(
            "participant_coordinates_checked",
            std::to_string(
                capacity_evidence.participant_coordinates_checked));
        capacity_tags.emplace(
            "tier_coordinates_checked",
            std::to_string(capacity_evidence.tier_coordinates_checked));
        capacity_tags.emplace(
            "malformed_edges",
            std::to_string(capacity_evidence.malformed_edges));
        capacity_tags.emplace(
            "participant_flow_violations",
            std::to_string(capacity_evidence.participant_flow_violations));
        capacity_tags.emplace(
            "tier_flow_violations",
            std::to_string(capacity_evidence.tier_flow_violations));
        capacity_tags.emplace(
            "direction_counts_are_capacity_proof",
            "false");
        /* Restoration still carries a full capacity proof, but it is a
         * teardown transaction rather than a live optimization wave. Keep its
         * evidence under a distinct name so diagnostics cannot inflate the
         * production movement/certification contract. */
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            prepared_context_restoration
                ? "prepared_context_restoration_capacity_conservation_certifications"
                : "capacity_conservation_certifications",
            1.0,
            prepared_context_restoration ? "model_teardown" : "maintenance",
            config_.perf_device,
            std::move(capacity_tags));
        if (transaction.economy.enabled)
        {
            add("committed_projected_service_gain_ns",
                static_cast<double>(
                    transaction.economy.projected_service_gain_ns));
            add("committed_projected_transfer_and_repack_ns",
                static_cast<double>(
                    transaction.economy
                        .projected_transfer_and_repack_ns));
            add("committed_projected_inference_interference_ns",
                static_cast<double>(
                    transaction.economy
                        .projected_inference_interference_ns));
            add("committed_projected_net_benefit_ns",
                static_cast<double>(
                    transaction.economy.projected_net_benefit_ns));
        }
        add("published_epoch", static_cast<double>(transaction.candidate->epoch));
    }

} // namespace llaminar2
