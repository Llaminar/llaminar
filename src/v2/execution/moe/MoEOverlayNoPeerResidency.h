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
            const MoERoutedExpertModelMetadata &model)
        {
            if (!hasNoPeerTopology(plan)) return std::nullopt;
            owners.requireLayerGeometry(model.num_layers, model.num_experts, model.first_model_layer);
            // Zero denotes unresolved construction metadata and cannot later
            // authorize a main-inference sampler boundary. A resolved endpoint
            // is retained verbatim; routed sidecar capacity is never its source.
            if (model.main_inference_layer_count < 0 ||
                (model.main_inference_layer_count != 0 &&
                 (model.main_inference_layer_count <= model.first_model_layer ||
                  static_cast<long long>(model.main_inference_layer_count) >
                      static_cast<long long>(model.first_model_layer) + model.num_layers)))
                throw std::invalid_argument("No-peer residency has an invalid declared main-inference endpoint");
            if (plan.first_model_layer != model.first_model_layer ||
                plan.placements.size() != static_cast<std::size_t>(model.num_layers) ||
                owners.participants().size() != 1u ||
                plan.authority_execution != resolveMoEOverlayAuthorityExecutionKind(plan))
                throw std::invalid_argument("No-peer residency requires the exact complete stage geometry");
            std::vector<bool> seen(static_cast<std::size_t>(model.num_layers));
            for (const auto &placement : plan.placements)
            {
                const auto row = static_cast<long long>(placement.layer) - model.first_model_layer;
                if (row < 0 || row >= model.num_layers || seen[static_cast<std::size_t>(row)] ||
                    placement.routed_expert_tier.size() != static_cast<std::size_t>(model.num_experts) ||
                    !std::all_of(placement.routed_expert_tier.begin(), placement.routed_expert_tier.end(),
                        [](int tier) { return tier == 0; }))
                    throw std::invalid_argument("No-peer residency has missing or foreign model membership");
                seen[static_cast<std::size_t>(row)] = true;
            }
            const auto &participant = owners.participants().front();
            const auto domain_rank = plan.domains.front().primaryWorldRank();
            if (!domain_rank || *domain_rank != participant.world_rank ||
                !participant.world_rank_known || participant.world_rank < 0 ||
                participant.address != plan.domains.front().participants.front() ||
                participant.domain_name != plan.domains.front().name || participant.tier_idx != 0 ||
                !participant.device.is_valid())
                throw std::invalid_argument("No-peer residency has a foreign or unresolved participant");
            for (const auto &owner : owners.owners())
                if (!owner.resident || owner.owner_participant != participant.participant_id ||
                    owner.device != participant.device || !owner.owner_world_rank_known ||
                    owner.owner_world_rank != participant.world_rank || owner.address != participant.address ||
                    owner.tier_idx != 0 || owner.domain_name != participant.domain_name)
                    throw std::invalid_argument("No-peer residency requires every expert's complete local payload");
            return MoEOverlayNoPeerResidency(participant.device, participant.world_rank,
                model.first_model_layer, model.num_layers, model.num_experts,
                model.main_inference_layer_count, plan.authority_execution);
        }

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
}
