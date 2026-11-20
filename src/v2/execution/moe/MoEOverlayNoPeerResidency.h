/**
 * @file MoEOverlayNoPeerResidency.h
 * @brief Immutable proof that one admitted stage owns every routed expert locally.
 *
 * This setup value authenticates a mathematical identity, not a disabled
 * Dynamic policy. A one-owner, one-tier stage has no alternative placement.
 * Its host/device route state remains owned by the original participant;
 * movement kernels, payload banks and collectives would perform no useful work.
 * The proof contains no live counts, placement mirror or memory ledger.
 */
#pragma once
#include "MoEExpertOwnerMap.h"
#include "MoEOptimizationStatus.h"
#include "MoERoutedExpertPlacementPlanner.h"
#include "execution/factory/FactoryPPStageConfig.h"
#include <optional>
#include <algorithm>
#include <vector>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Fully checked immutable local-residency identity for a PP stage. */
    class MoEOverlayNoPeerResidency final
    {
    public:
        /**
         * @brief Recognize only an explicit one-tier/one-domain/one-participant shape.
         * @param plan Normalized immutable plan, possibly before quota resolution.
         * @return Whether no second routed placement endpoint exists.
         *
         * This is geometry only. No execution exemption is valid until seal()
         * authenticates every physical resident in the model-frozen owner map.
         */
        [[nodiscard]] static bool hasNoPeerTopology(const MoERoutedExpertPlacementPlan &plan) noexcept
        {
            return plan.usesExpertOverlayAuthority() && plan.routed_tiers.size() == 1u &&
                plan.domains.size() == 1u && plan.domains.front().participants.size() == 1u &&
                plan.routed_tiers.front().domain == plan.domains.front().name &&
                plan.effectiveBaseModelDomain() == plan.domains.front().name;
        }

        /**
         * @brief Authenticate complete residency after the sole initial map is frozen.
         * @param plan Exact initial plan retained by the residency authority.
         * @param owners Its complete immutable physical owner map.
         * @param model Exact global layer interval and routed expert count.
         * @return A proof for singleton topology, or no exemption for other shapes.
         * @throws std::invalid_argument for incomplete, nonresident or foreign singleton ownership,
         *         or a malformed nonzero canonical main endpoint. Zero remains construction-only metadata.
         */
        [[nodiscard]] static std::optional<MoEOverlayNoPeerResidency> seal(
            const MoERoutedExpertPlacementPlan &plan, const MoEExpertOwnerMap &owners,
            const MoERoutedExpertModelMetadata &model);

        /**
         * @brief Reject attaching this identity to a different graph participant.
         * @param device Exact participant-local execution device.
         * @param first Global first layer owned by the graph's routed runtime.
         * @param layers Complete owned routed layer count, including admitted sidecars.
         * @param experts Routed expert count for each owned layer.
         * @throws std::invalid_argument for changed device or geometry.
         */
        void requireDestination(DeviceId device, int first, int layers, int experts) const
        {
            if (device != device_ || first != first_ || layers != layers_ || experts != experts_)
                throw std::invalid_argument("No-peer identity belongs to another stage participant");
        }
        /**
         * @brief Authenticate the only overlay role imported into a native PP tail parent.
         * @param proof Original canonical authority proof, absent for movable domains.
         * @param stage Exact terminal main-layer scope; admitted sidecar banks may follow it.
         * @param device Participant-local sampler/epoch device.
         * @param experts Complete routed expert count of the graph.
         * @throws std::invalid_argument For missing proof, foreign scope/device or a non-tail role.
         *
         * This checks immutable topology only. The ordinary materializer still
         * authenticates the live admission, epoch arena and native recording;
         * this value cannot manufacture a sampler or execution capability.
         */
        static void requireNativePipelineTail(
            const std::optional<MoEOverlayNoPeerResidency> &proof,
            const FactoryPPStageConfig &stage, DeviceId device, int experts)
        {
            if (!proof || !device.is_gpu() || !stage.has_lm_head || stage.has_embedding ||
                stage.first_layer <= 0 || stage.last_layer <= stage.first_layer ||
                stage.first_layer != proof->first_ ||
                proof->main_end_ <= proof->first_ || stage.last_layer != proof->main_end_ ||
                proof->authority_ != MoEOverlayAuthorityExecutionKind::DeviceResident)
                throw std::invalid_argument("Native pipeline overlay sampler requires the exact admitted no-peer GPU tail");
            proof->requireDestination(device, stage.first_layer, proof->layers_, experts);
        }

        /** @return Frozen authority locus, independent of whether movement can occur. */
        [[nodiscard]] MoEOverlayAuthorityExecutionKind authorityExecution() const noexcept { return authority_; }
        /** @return Sole physical execution device; not a transfer destination selector. */
        [[nodiscard]] DeviceId device() const noexcept { return device_; }
        /** @return Resolved process owner used by the original admission. */
        [[nodiscard]] int worldRank() const noexcept { return world_rank_; }
        /**
         * @brief Publish a passive zero-work receipt without inventing completed decisions.
         * @return Original authority with a precise topology-only nonmoving reason.
         *
         * MovementDisabled describes impossible movement, not a rewrite of the
         * requested Dynamic policy. No histogram window, command or byte is
         * claimed to have run. Native route execution remains independently
         * certified by its captured serving graph.
         */
        [[nodiscard]] MoEOptimizationStatus status() const
        {
            return {.authority = authority_ == MoEOverlayAuthorityExecutionKind::DeviceResident
                        ? MoEOptimizationAuthority::Device : MoEOptimizationAuthority::Host,
                .state = MoEOptimizationLifecycleState::MovementDisabled,
                .activity = MoEOptimizationActivityState::Dormant,
                .diagnostic = "Authenticated no-peer topology; original maintenance policy retained"};
        }
    private:
        /** @brief Only seal() can mint a complete local-residency proof. */
        MoEOverlayNoPeerResidency(DeviceId device, int rank, int first, int layers, int experts,
            int main_end, MoEOverlayAuthorityExecutionKind authority)
            : device_(device), world_rank_(rank), first_(first), layers_(layers), experts_(experts),
              main_end_(main_end), authority_(authority) {}
        DeviceId device_;
        int world_rank_, first_, layers_, experts_;
        /** Canonical exclusive main endpoint; zero is explicitly unresolved metadata. */
        int main_end_;
        MoEOverlayAuthorityExecutionKind authority_;
    };
    /**
     * @brief Recognize the only routed target of a homogeneous GPU shard cohort.
     *
     * This shape excludes movement storage at capacity admission. The runtime
     * separately authenticates complete resident source rows after quotas seal.
     */
    [[nodiscard]] inline bool hasStationaryTensorShardTopology(
        const MoERoutedExpertPlacementPlan &plan) noexcept
    {
        if (!plan.usesExpertOverlayAuthority() || plan.routed_tiers.size() != 1 ||
            plan.routed_tiers.front().domain != plan.continuation_domain ||
            plan.shared_expert_domain != plan.continuation_domain)
            return false;
        const auto domain = std::find_if(plan.domains.begin(), plan.domains.end(),
            [&](const auto &candidate) { return candidate.name == plan.continuation_domain; });
        if (domain == plan.domains.end() || domain->participants.size() < 2 ||
            domain->scope != ExecutionDomainScope::RANK_LOCAL || domain->owner_rank < 0 ||
            domain->routed_compute_policy != RoutedExpertComputePolicy::TensorSharded ||
            domain->routed_phase_policy != RoutedExpertPhasePolicy::Uniform ||
            domain->routed_decode_assignment_policy != RoutedExpertAssignmentPolicy::StaticOwner ||
            domain->routed_prefill_assignment_policy != RoutedExpertAssignmentPolicy::StaticOwner)
            return false;
        const auto backend = domain->participants.front().device_type;
        return std::all_of(domain->participants.begin(), domain->participants.end(), [&](const auto &participant) {
            return participant.isGPU() && participant.device_type == backend;
        });
    }

    /** @brief Authenticate complete original sharded residency with no alternate routed target. */
    class MoEOverlayStationaryTensorShardResidency final
    {
    public:
        /** @return A proof for an exact, fully resident single sharded cohort, or no proof for other shapes. */
        [[nodiscard]] static std::optional<MoEOverlayStationaryTensorShardResidency> seal(
            const MoERoutedExpertPlacementPlan &plan, const MoEExpertOwnerMap &owners,
            const MoERoutedExpertModelMetadata &model);
        /**
         * @brief Project zero movement once from the original continuation root.
         * @param device Exact graph participant whose status is requested.
         * @return The root's stationary status or an empty follower status.
         * @throws std::invalid_argument For a foreign device.
         */
        [[nodiscard]] MoEOptimizationStatus statusFor(DeviceId device) const;
    private:
        /** @brief Store the admitted participant order and its declared logical root. */
        MoEOverlayStationaryTensorShardResidency(std::vector<DeviceId> participants,
            DeviceId root, MoEOptimizationAuthority authority);
        std::vector<DeviceId> participants_;
        DeviceId root_;
        MoEOptimizationAuthority authority_;
    };

}
