/**
 * @file MoEExpertOwnerMap.h
 * @brief Graph-native routed-MoE whole-expert ownership authority.
 *
 * A placement plan names the tier containing every expert; this type resolves
 * that logical placement to exactly one physical participant. Initial maps use
 * the configured ordinal or deterministic-random order. Epoch transitions use
 * the prior immutable map as a stability constraint so experts that remain in
 * a multi-participant tier do not move merely because another expert entered
 * or left that tier.
 */

#pragma once

#include "MoELayeredExpertOwnership.h"
#include "MoERoutedExpertPlacementPlan.h"
#include "backends/DeviceId.h"
#include "backends/GlobalDeviceAddress.h"

#include <cstddef>
#include <string>
#include <vector>

namespace llaminar2
{
    /** @brief Physical owner and residency metadata for one routed expert. */
    struct MoEExpertOwner
    {
        int layer_idx = -1;
        int expert_id = -1;
        int tier_idx = -1;
        int owner_participant = -1;
        DeviceId device = DeviceId::cpu();
        bool resident = false;

        std::string tier_name;
        std::string domain_name;
        int domain_participant_index = -1;
        int owner_world_rank = -1;
        bool owner_world_rank_known = false;
        GlobalDeviceAddress address;
    };

    /** @brief Stable participant descriptor referenced by expert owners. */
    struct MoEExpertOwnerParticipant
    {
        int participant_id = -1;
        int tier_idx = -1;
        std::string tier_name;
        std::string domain_name;
        int domain_participant_index = -1;
        GlobalDeviceAddress address;
        DeviceId device = DeviceId::invalid();
        int world_rank = -1;
        bool world_rank_known = false;
    };

    /** @brief Validation policy for whole-expert owner-map construction. */
    struct MoEExpertOwnerMapBuildOptions
    {
        /// A whole-expert owner map cannot describe multiple tensor-shard
        /// owners unless the caller is deliberately using it only as a
        /// placement/residency index.
        bool reject_tensor_sharded_domains = true;
    };

    /**
     * @brief Complete `(layer, expert) -> participant` ownership relation.
     *
     * The map is immutable after construction and is safe to publish as part
     * of an ExpertOverlay residency epoch. Participant identifiers remain
     * stable only while the routed-tier topology is unchanged.
     */
    class MoEExpertOwnerMap
    {
    public:
        /**
         * @brief Build deterministic ownership for an initial placement.
         * @param plan Valid tiered whole-expert placement plan.
         * @param options Whole-expert validation policy.
         * @return Complete balanced owner map.
         * @throws std::invalid_argument for invalid placement or topology.
         */
        static MoEExpertOwnerMap build(
            const MoERoutedExpertPlacementPlan &plan,
            const MoEExpertOwnerMapBuildOptions &options = {});

        /**
         * @brief Build a balanced epoch transition with stable retained owners.
         *
         * Experts whose tier is unchanged keep their previous participant up
         * to that participant's exact balanced target count. Only vacated or
         * unavoidable deficit slots are assigned using the plan's configured
         * owner order. This makes physical movement proportional to the logical
         * tier transition instead of to incidental list re-partitioning.
         *
         * @param plan Valid candidate placement using the same topology/model.
         * @param previous Complete immutable owner map for the prior epoch.
         * @param options Whole-expert validation policy.
         * @return Complete balanced owner map for the candidate epoch.
         * @throws std::invalid_argument when topology or model geometry differs.
         */
        static MoEExpertOwnerMap buildTransition(
            const MoERoutedExpertPlacementPlan &plan,
            const MoEExpertOwnerMap &previous,
            const MoEExpertOwnerMapBuildOptions &options = {});

        /**
         * @brief Materialize an explicitly planned physical participant map.
         *
         * Tier placement and participant placement are separate optimization
         * axes. Histogram-driven ExpertOverlay maintenance may keep an expert
         * in the same tier while swapping its exact owner to reduce domain
         * skew. Re-deriving ownership from ordinal/random cold-start order
         * would erase that decision, so an RCU candidate uses this constructor
         * after its participant planner has produced a complete dense table.
         *
         * Every explicit owner must be a participant of the expert's selected
         * tier. The method validates topology and totality but deliberately
         * does not compare capacities with a prior epoch; the residency
         * authority performs that transition-level invariant check.
         *
         * @param plan Complete logical tier placement for the candidate epoch.
         * @param ownership Exact `(layer, expert) -> participant` assignment.
         * @param options Whole-expert validation policy.
         * @return Complete immutable owner map using the explicit assignment.
         * @throws std::invalid_argument for incompatible geometry or tier owners.
         */
        static MoEExpertOwnerMap buildExplicit(
            const MoERoutedExpertPlacementPlan &plan,
            const MoELayeredExpertOwnership &ownership,
            const MoEExpertOwnerMapBuildOptions &options = {});

        /**
         * @return Every physical owner in canonical `(layer, expert)` order.
         *
         * Initial, transition, and explicit construction all expose the same
         * order.  Distributed identity and migration code may therefore
         * consume this vector without accidentally hashing the construction
         * algorithm which produced an otherwise identical owner relation.
         */
        const std::vector<MoEExpertOwner> &owners() const { return owners_; }
        /** @return Stable participant descriptors indexed by participant ID. */
        const std::vector<MoEExpertOwnerParticipant> &participants() const { return participants_; }

        /**
         * @brief Look up one immutable owner through the canonical coordinate index.
         * @param layer_idx Exact transformer-layer coordinate.
         * @param expert_id Exact routed-expert coordinate within that layer.
         * @return Owner for the model expert, or null when it is absent.
         *
         * Construction sorts the complete relation by `(layer, expert)`, so
         * this graph-facing query is logarithmic rather than scanning every
         * expert in the model for each routed activation.
         */
        const MoEExpertOwner *ownerFor(int layer_idx, int expert_id) const;
        /** @return Descriptor for a participant ID, or null when absent. */
        const MoEExpertOwnerParticipant *participantForId(int participant_id) const;

        /** @return Participant IDs belonging to `tier_idx`, in stable order. */
        std::vector<int> participantIdsForTier(int tier_idx) const;
        /** @return Sorted expert IDs owned by one participant in one layer. */
        std::vector<int> expertsForParticipant(int layer_idx, int owner_participant) const;
        /** @return Dense expert mask equivalent to `expertsForParticipant()`. */
        std::vector<bool> expertMaskForParticipant(
            int layer_idx,
            int owner_participant,
            int num_experts) const;
        /**
         * @brief Report total ownership for one model coordinate.
         * @return Zero for an absent coordinate, otherwise exactly one.
         *
         * Duplicate ownership is rejected during construction, which makes a
         * larger result unrepresentable in every live map.
         */
        size_t ownerCountForExpert(int layer_idx, int expert_id) const;

        /**
         * @brief Materialize the complete layered ownership table represented here.
         *
         * Residency planning, histogram attribution, and runtime-bank publication
         * must consume one identical `(layer, expert) -> participant` relation.
         * Keeping this conversion with the owner-map authority avoids each caller
         * rebuilding the dense table with subtly different completeness checks.
         *
         * @param num_layers Exact model transformer-layer count.
         * @param num_experts Exact routed-expert count per layer.
         * @return Validated dense ownership indexed by layer and expert.
         * @throws std::invalid_argument for non-positive geometry or no participants.
         * @throws std::logic_error when an owner is missing, duplicated, or outside
         *         the requested model geometry.
         */
        [[nodiscard]] MoELayeredExpertOwnership layeredOwnership(
            int num_layers,
            int num_experts) const;

    private:
        /** @brief Shared implementation for initial and transition builds. */
        static MoEExpertOwnerMap buildWithPreferredOwners(
            const MoERoutedExpertPlacementPlan &plan,
            const MoEExpertOwnerMapBuildOptions &options,
            const MoEExpertOwnerMap *previous);

        std::vector<MoEExpertOwner> owners_;
        std::vector<MoEExpertOwnerParticipant> participants_;
    };

} // namespace llaminar2
