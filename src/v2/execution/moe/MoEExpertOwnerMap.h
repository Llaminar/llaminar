/**
 * @file MoEExpertOwnerMap.h
 * @brief Canonical routed-MoE whole-owner and source-bound shard-cohort placement authority.
 *
 * A placement plan names the tier containing every expert; this type resolves
 * that logical placement to one whole owner or an explicit joint fragment cohort.
 * Ordinary graph execution accessors reject joint placement until its real
 * prepared/collective path exists; no participant is chosen as a false whole
 * owner. Initial whole-expert maps use
 * the configured ordinal or deterministic-random order. Epoch transitions use
 * the prior immutable map as a stability constraint so experts that remain in
 * a multi-participant tier do not move merely because another expert entered
 * or left that tier. Explicit whole-owner participant changes remain checked
 * against that same canonical map while retaining every CPU cohort intact.
 */

#pragma once

#include "MoELayeredExpertOwnership.h"
#include <memory>
#include "MoERoutedExpertPlacementPlan.h"
#include "backends/DeviceId.h"
#include "backends/GlobalDeviceAddress.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <utility>
#include <vector>

namespace llaminar2
{
    class MoEExpertShardCohort;
    struct MoEOverlayLayerWeightManifest;
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

        /** Joint fragment alternative; when present no single whole-expert owner exists. */
        std::shared_ptr<const MoEExpertShardCohort> shard_cohort;

        /** @param other Original logical placement and physical fragment definition.
         * @return Structural metadata equality, independent of a shared cohort wrapper address.
         * @note Original source liveness and native prepared authority remain separate guards. */
        bool operator==(const MoEExpertOwner &other) const;
    };

    /**
     * @brief One physical placement vertex: a whole owner or a complete shard cohort.
     *
     * This immutable value is derived from the canonical owner row. It owns no
     * capacity or live placement ledger. A cohort consumes one fragment slot at
     * every original member, but remains one logical expert at the tier. Keeping
     * the alternatives typed prevents the cohort's absent scalar owner (-1)
     * from accidentally becoming a physical endpoint or joining unrelated cycles.
     */
    class MoEExpertPlacementTarget final
    {
    public:
        /**
         * @param placement Original whole-owner or source-bound cohort row.
         * @return A checked target, or no value for malformed/ambiguous ownership.
         * @note No allocation, source loading or execution admission occurs here.
         */
        [[nodiscard]] static std::optional<MoEExpertPlacementTarget> fromPlacement(
            const MoEExpertOwner &placement) noexcept;

        /** @return Original logical tier coordinate, independent of cohort width. */
        [[nodiscard]] int tierIndex() const noexcept { return tier_idx_; }
        /** @return Whether every original fragment member jointly owns the expert. */
        [[nodiscard]] bool isCohort() const noexcept;
        /** @return One for a whole owner, otherwise the complete original cohort width. */
        [[nodiscard]] std::size_t participantCount() const noexcept;
        /**
         * @param index Original fixed cohort order, or zero for a whole owner.
         * @return Exact canonical physical participant ID.
         * @throws std::out_of_range For an index outside this complete target.
         */
        [[nodiscard]] int participantId(std::size_t index) const;
        /**
         * @param other Another placement vertex in the same canonical topology.
         * @return Equality of tier, members and original source/cuts; expert ID is
         * deliberately absent because different experts occupy the same slots.
         */
        [[nodiscard]] bool operator==(const MoEExpertPlacementTarget &other) const noexcept;

    private:
        using Owners = std::variant<int, std::shared_ptr<const MoEExpertShardCohort>>;
        /**
         * @brief Retain only a validated original tier and ownership alternative.
         * @param tier_idx Canonical logical tier coordinate.
         * @param owners Whole participant or complete immutable original cohort.
         */
        MoEExpertPlacementTarget(int tier_idx, Owners owners) noexcept
            : tier_idx_(tier_idx), owners_(std::move(owners)) {}
        int tier_idx_;
        Owners owners_;
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

        /** @return Equality of the complete participant identity, including rank. */
        bool operator==(const MoEExpertOwnerParticipant &) const = default;
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
        /** @brief Resolve only canonical authored participants before quota/ownership selection.
         * @param plan Existing routed tier/domain topology, including empty expert endpoints.
         * @param options Same domain validation used by actual owner-map construction.
         * @return Stable descriptors with the exact IDs later used by initial/transition maps.
         * @note This constructs no expert ownership rows and depends on no tier capacities.
         */
        static std::vector<MoEExpertOwnerParticipant> declaredParticipants(
            const MoERoutedExpertPlacementPlan &plan,const MoEExpertOwnerMapBuildOptions &options = {});

        /**
         * @brief Classify actual multi-member CPU tensor fragments without constructing owners or quotas.
         * @param plan Original tier/domain declaration; unused domains do not create expert participants.
         * @return True only for a referenced tensor-sharded CPU domain with more than one member.
         * @note Explicit CPU TP1 retains its policy identity and uses full-matrix ordinary storage.
         */
        [[nodiscard]] static bool usesCPUJointFragments(const MoERoutedExpertPlacementPlan& plan) noexcept;

        /**
         * @brief Observe actual cold canonical constructor entries, including failures.
         * @return Monotonic process-wide count; scoped deltas prove construction
         * economy without a reset, source mirror or alternate owner authority.
         * This observation never selects placement or execution policy.
         */
        [[nodiscard]] static std::uint64_t initialBuildAttempts() noexcept;

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
         * @brief Resolve source-native CPU shard cohorts in the canonical placement authority.
         * @param plan Complete original logical tier placements and bound CPU node-TP membership.
         * @param manifest Exact original owned layer source contracts, retaining global IDs.
         * @return One logical placement row per expert, with either a whole owner or a joint cohort.
         * @throws std::invalid_argument For missing/foreign source, endpoints or unsupported native cuts.
         * @note This is placement only. Ordinary whole-expert execution access remains rejecting.
         */
        static MoEExpertOwnerMap buildShardPlacement(const MoERoutedExpertPlacementPlan& plan,
            const std::vector<MoEOverlayLayerWeightManifest>& manifest);

        /**
         * @brief Derive a capacity-preserving live placement with fixed original CPU cohorts.
         * @param plan Candidate tier assignments with the unchanged declared topology.
         * @param previous Real previous live placement, retaining stable whole owners.
         * @param original Immutable source placement owning the original cohort/cut identities.
         * @param whole_owner_changes Explicit whole-owner participant changes relative to
         *        the canonical tier transition: previous_participant names the
         *        post-tier-projection owner, including newly promoted experts.
         *        Empty retains its stable owners; explicit no-op edges are rejected.
         * @return One canonical live map; no source selection or physical readiness is minted.
         * @throws std::invalid_argument For foreign topology, geometry, quotas or cohort
         * identities, stale/duplicate changes, non-apportioned or foreign whole destinations,
         * or a change in per-layer participant capacity.
         * @note Original cohorts are shared, never rebuilt from candidate expert IDs.
         * Participant changes cannot name a cohort member as a whole owner. Every
         * changed whole-owner slot is paired within its original layer/tier before publication.
         */
        static MoEExpertOwnerMap buildShardTransition(const MoERoutedExpertPlacementPlan &plan,
                                                      const MoEExpertOwnerMap &previous,
                                                      const MoEExpertOwnerMap &original,
                                                      std::span<const MoELayeredExpertOwnershipChange> whole_owner_changes = {});

        /**
         * @param layer_idx Original model-global layer.
         * @param expert_id Original expert ID.
         * @return The canonical logical placement, explicitly including any shard cohort, or null.
         */
        const MoEExpertOwner* placementFor(int layer_idx,int expert_id) const;

        /**
         * @brief Borrow one layer's complete canonical placement relation.
         * @param layer_idx Model-global layer, including sparse or absent IDs.
         * @return Ordered original rows, including whole owners and cohorts;
         *         empty for an absent layer. The immutable map owns the lifetime.
         * @note Every builder canonicalizes `(layer, expert)` once before
         *       publication. Two binary searches retain this invariant without
         *       constructing a second index or scanning other model layers.
         */
        [[nodiscard]] std::span<const MoEExpertOwner> placementsForLayer(int layer_idx) const noexcept;

        /**
         * @param layer_idx Original model-global layer.
         * @param expert_id Original expert ID.
         * @return Exact physical fragment count; zero if absent, one for a whole owner.
         */
        std::size_t fragmentCountForExpert(int layer_idx,int expert_id) const;

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
         * @return Whole owner for the model expert, or null when it is absent.
         * @throws std::logic_error When the placement is jointly tensor-sharded; no whole-owner alias exists.
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
         * @throws std::logic_error For a joint shard placement: use fragmentCountForExpert.
         * Duplicate ownership is rejected during construction, which makes a
         * larger result unrepresentable in every live map.
         */
        size_t ownerCountForExpert(int layer_idx, int expert_id) const;

        /**
         * @brief Authenticate whole or complete-cohort geometry without allocating a second table.
         * @param num_layers Number of owned contiguous layers.
         * @param num_experts Routed experts per owned layer.
         * @param first_model_layer First global model layer in the interval.
         * @throws std::invalid_argument For invalid or overflowing geometry.
         * @throws std::logic_error For incomplete or foreign ownership coordinates.
         * Every fragment must retain its original physical rank/NUMA member;
         * success does not make a jointly owned expert a scalar whole owner.
         */
        void requireLayerGeometry(int num_layers, int num_experts, int first_model_layer = 0) const;

        /**
         * @brief Materialize the complete layered ownership table represented here.
         *
         * Residency planning, histogram attribution, and runtime-bank publication
         * must consume one identical `(layer, expert) -> participant` relation.
         * Keeping this conversion with the owner-map authority avoids each caller
         * rebuilding the dense table with subtly different completeness checks.
         *
         * @param num_layers Exact number of owned contiguous model layers.
         * @param num_experts Exact routed-expert count per layer.
         * @param first_model_layer First model-global layer in the owned interval.
         * @return Compact ownership retaining model-global layer identities.
         * @throws std::logic_error If any expert requires a shard cohort instead of one owner.
         * @throws std::invalid_argument for non-positive geometry or no participants.
         * @throws std::logic_error when an owner is missing, duplicated, or outside
         *         the requested model geometry.
         */
        [[nodiscard]] MoELayeredExpertOwnership layeredOwnership(
            int num_layers,
            int num_experts,
            int first_model_layer = 0) const;

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
