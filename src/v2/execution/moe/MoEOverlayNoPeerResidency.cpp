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
    std::optional<MoEOverlayStationaryTensorShardResidency>
    MoEOverlayStationaryTensorShardResidency::seal(
        const MoERoutedExpertPlacementPlan &plan, const MoEExpertOwnerMap &owners,
        const MoERoutedExpertModelMetadata &model)
    {
        if (!hasStationaryTensorShardTopology(plan)) return std::nullopt;
        owners.requireLayerGeometry(model.num_layers, model.num_experts, model.first_model_layer);
        const auto domain = std::find_if(plan.domains.begin(), plan.domains.end(),
            [&](const auto &candidate) { return candidate.name == plan.continuation_domain; });
        if (plan.first_model_layer != model.first_model_layer ||
            plan.placements.size() != static_cast<std::size_t>(model.num_layers) ||
            owners.participants().size() != domain->participants.size() ||
            plan.authority_execution != resolveMoEOverlayAuthorityExecutionKind(plan))
            throw std::invalid_argument("Stationary tensor-sharded residency requires the complete original source scope");
        std::vector<bool> layers(static_cast<size_t>(model.num_layers));
        for (const auto &placement : plan.placements)
        {
            const auto offset = static_cast<long long>(placement.layer) - model.first_model_layer;
            if (offset < 0 || offset >= model.num_layers || layers[static_cast<size_t>(offset)] ||
                placement.routed_expert_tier.size() != static_cast<size_t>(model.num_experts) ||
                !std::all_of(placement.routed_expert_tier.begin(), placement.routed_expert_tier.end(),
                    [](int tier) { return tier == 0; }))
                throw std::invalid_argument("Stationary tensor-sharded residency cannot omit or move any logical expert");
            layers[static_cast<size_t>(offset)] = true;
        }
        for (const auto &owner : owners.owners())
        {
            if (!owner.resident || owner.tier_idx != 0 || owner.domain_name != domain->name ||
                !owner.shard_cohort || owner.owner_participant != -1 || owner.device.is_valid() ||
                owner.shard_cohort->source().layer_idx != owner.layer_idx ||
                owner.shard_cohort->participants().size() != domain->participants.size())
                throw std::invalid_argument("Stationary tensor-sharded residency requires every original expert cohort");
            const auto &cohort = *owner.shard_cohort;
            for (size_t index = 0; index < cohort.participants().size(); ++index)
            {
                const auto &member = cohort.participants()[index];
                const auto *participant = owners.participantForId(member.participant_id);
                const auto &geometry = cohort.ownership(index).geometry();
                if (!participant || participant->tier_idx != 0 ||
                    participant->domain_participant_index != static_cast<int>(index) ||
                    participant->domain_name != domain->name ||
                    !participant->world_rank_known || participant->world_rank != domain->owner_rank ||
                    member.world_rank != domain->owner_rank ||
                    participant->address != member.address || member.address != domain->participants[index] ||
                    geometry.experts != model.num_experts || geometry.model_columns != model.d_model ||
                    geometry.intermediate_columns != model.routed_intermediate_size)
                    throw std::invalid_argument("Stationary tensor-sharded residency changed original rank, source or member order");
            }
        }
        const auto root = plan.continuation_domain_spec.logical_root_participant;
        if (root < 0 || static_cast<std::size_t>(root) >= domain->participants.size())
            throw std::invalid_argument("Stationary tensor-shard evidence lost its logical continuation root");
        const auto root_device = domain->participants[static_cast<std::size_t>(root)].toLocalDeviceId();
        std::vector<DeviceId> participants;
        participants.reserve(domain->participants.size());
        for (const auto &address : domain->participants)
            participants.push_back(address.toLocalDeviceId());
        return MoEOverlayStationaryTensorShardResidency(std::move(participants), root_device,
            plan.authority_execution == MoEOverlayAuthorityExecutionKind::DeviceResident
                ? MoEOptimizationAuthority::Device : MoEOptimizationAuthority::Host);
    }

    MoEOverlayStationaryTensorShardResidency::MoEOverlayStationaryTensorShardResidency(
        std::vector<DeviceId> participants, DeviceId root, MoEOptimizationAuthority authority)
        : participants_(std::move(participants)), root_(root), authority_(authority)
    {
    }

    MoEOptimizationStatus MoEOverlayStationaryTensorShardResidency::statusFor(DeviceId device) const
    {
        if (std::find(participants_.begin(), participants_.end(), device) == participants_.end())
            throw std::invalid_argument("Stationary tensor-shard evidence requested by a foreign device");
        if (device != root_)
            return {};
        return {.authority = authority_,
            .state = MoEOptimizationLifecycleState::MovementDisabled,
            .activity = MoEOptimizationActivityState::Dormant,
            .diagnostic = "Single fully resident tensor-sharded cohort has no alternate routed target"};
    }

}
