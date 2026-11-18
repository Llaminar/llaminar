/**
 * @file MoEOverlayNoPeerResidency.cpp
 * @brief Authenticate complete original singleton ownership at the source boundary.
 *
 * Cohort source geometry stays out of the shared authority header. Keeping this
 * setup-only implementation separate avoids a dependency cycle through physical
 * migration while preserving the exact same immutable no-peer proof.
 */
#include "MoEOverlayNoPeerResidency.h"
#include "MoEExpertShardCohort.h"

namespace llaminar2
{
    std::optional<MoEOverlayNoPeerResidency> MoEOverlayNoPeerResidency::seal(
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
        {
            if(!owner.resident || owner.tier_idx!=0 || owner.domain_name!=participant.domain_name)
                throw std::invalid_argument("No-peer residency requires every expert's complete local payload");
            if(owner.shard_cohort) {
                // GPU1 remains its original cohort arm: no scalar owner or
                // device alias is minted to authorize the no-peer identity.
                const auto& cohort=*owner.shard_cohort;
                if(!participant.device.is_gpu() || cohort.participants().size()!=1u ||
                   owner.owner_participant!=-1 || owner.device.is_valid() ||
                   cohort.source().layer_idx!=owner.layer_idx)
                    throw std::invalid_argument("No-peer residency changed its original singleton GPU cohort");
                const auto& member=cohort.participants().front();
                const auto& layout=cohort.ownership(0);const auto& geometry=layout.geometry();
                if(member.participant_id!=participant.participant_id || member.world_rank!=participant.world_rank ||
                   member.address!=participant.address || geometry.experts!=model.num_experts ||
                   geometry.model_columns!=model.d_model || geometry.intermediate_columns!=model.routed_intermediate_size)
                    throw std::invalid_argument("No-peer residency changed its original complete GPU source geometry");
                for(const auto role:{WeightRole::MoEExpertGate,WeightRole::MoEExpertUp,WeightRole::MoEExpertDown}) {
                    const auto projection=layout.projection(role);
                    if(projection.first_row!=0 || projection.first_column!=0 ||
                       projection.rows!=projection.source_rows || projection.columns!=projection.source_columns)
                        throw std::invalid_argument("No-peer GPU cohort requires its original full source projections");
                }
            } else if(owner.owner_participant!=participant.participant_id || owner.device!=participant.device ||
                !owner.owner_world_rank_known || owner.owner_world_rank!=participant.world_rank || owner.address!=participant.address)
                throw std::invalid_argument("No-peer residency requires every expert's complete local payload");
        }
        return MoEOverlayNoPeerResidency(participant.device, participant.world_rank,
            model.first_model_layer, model.num_layers, model.num_experts,
            model.main_inference_layer_count, plan.authority_execution);
    }
}
