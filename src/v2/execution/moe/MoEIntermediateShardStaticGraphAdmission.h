/**
 * @file MoEIntermediateShardStaticGraphAdmission.h
 * @brief Authenticate the original homogeneous GPU domain at static model graph admission.
 *
 * This setup-only check borrows the original source owner map and native TP
 * context. It neither admits memory nor creates a communicator or source. It
 * narrows the unfinished model boundary to one static domain; follower service,
 * moving pools and full serving qualification remain separate requirements.
 */
#pragma once
#include "MoEExpertOwnerMap.h"
#include "MoEExpertShardCohort.h"
#include "collective/ILocalTPContext.h"
#include "execution/config/RuntimeConfig.h"
#include <algorithm>
#include <stdexcept>
namespace llaminar2
{
/**
 * @brief Recognize the exact two-tier original GPU LocalTP + CPU node-local sharded service.
 *
 * The per-rank original source and mandatory transport are authenticated by
 * graph admission after this shape predicate.
 */
inline bool hasMappedGPUCPUIntermediateShardTopology(const MoERoutedExpertPlacementPlan &plan)
{
    if (!plan.usesExpertOverlayAuthority() || plan.routed_tiers.size() != 2 ||
        (plan.authority_execution != MoEOverlayAuthorityExecutionKind::HostResident &&
         plan.authority_execution != MoEOverlayAuthorityExecutionKind::Unresolved) ||
        plan.shared_expert_domain != plan.continuation_domain ||
        plan.routed_tiers.front().domain != plan.continuation_domain ||
        plan.routed_tiers.back().domain == plan.continuation_domain)
        return false;
    const auto gpu = std::find_if(plan.domains.begin(), plan.domains.end(),
        [&](const auto &domain) { return domain.name == plan.continuation_domain; });
    const auto cpu = std::find_if(plan.domains.begin(), plan.domains.end(),
        [&](const auto &domain) { return domain.name == plan.routed_tiers.back().domain; });
    const auto uniformShard = [](const auto &domain) {
        return domain.routed_compute_policy == RoutedExpertComputePolicy::TensorSharded &&
               domain.routed_phase_policy == RoutedExpertPhasePolicy::Uniform &&
               domain.routed_decode_assignment_policy == RoutedExpertAssignmentPolicy::StaticOwner &&
               domain.routed_prefill_assignment_policy == RoutedExpertAssignmentPolicy::StaticOwner;
    };
    if (gpu == plan.domains.end() || cpu == plan.domains.end() ||
        gpu->scope != ExecutionDomainScope::RANK_LOCAL || cpu->scope != ExecutionDomainScope::NODE_LOCAL ||
        gpu->participants.size() < 2 || cpu->participants.size() < 2 || !uniformShard(*gpu) || !uniformShard(*cpu))
        return false;
    const auto backend = gpu->participants.front().device_type;
    return std::all_of(gpu->participants.begin(), gpu->participants.end(), [&](const auto &member) {
            return member.isGPU() && member.device_type == backend;
        }) &&
        std::all_of(cpu->participants.begin(), cpu->participants.end(), [](const auto &member) { return member.isCPU(); });
}

/**
 * @brief Check maintenance support of a GPU intermediate-shard graph.
 * @param mode Original serving maintenance request.
 * @param mapped_ticket Whether the admitted topology has a mapped GPU/CPU ticket owner.
 * @return Whether a producer exists for the requested maintenance policy.
 *
 * Only the mapped continuation owns an executed-row observation ticket.
 * Dynamic still requires the original stationary, no-peer, or mapped authority.
 */
[[nodiscard]] constexpr bool supportsGPUIntermediateShardMaintenance(
    MoERebalanceRuntimeMode mode, bool mapped_ticket) noexcept
{
    return mode == MoERebalanceRuntimeMode::Off ||
           mode == MoERebalanceRuntimeMode::Dynamic ||
           (mode == MoERebalanceRuntimeMode::Observe && mapped_ticket);
}

/**
 * @brief Require exact original GPU source members, source-layer scope and native collective order.
 * @param plan Original authored source placement, without an execution-policy substitution.
 * @param owners Same immutable joint source map retained by preparation/residency.
 * @param context Original native local TP context; a singleton HOST identity performs no transport.
 * @param device Actual graph GPU, never a logical member coordinate.
 * @param member Original domain-local coordinate.
 * @param world_rank Original process owner of this RANK_LOCAL domain.
 * @param first_layer Original global source-layer origin.
 * @param layer_count Complete retained main/predictor source-layer count.
 * @param experts Original complete logical expert count.
 * @return Borrowed original participant; no graph, source reader or native resource is retained.
 * @throws std::invalid_argument For missing/foreign source, collective, rank, order or scope.
 */
inline const MoEExpertOwnerParticipant &requireOriginalGPUIntermediateShardGraph(
    const MoERoutedExpertPlacementPlan &plan, const MoEExpertOwnerMap &owners,
    const ILocalTPContext &context, DeviceId device, int member, int world_rank,
    int first_layer, int layer_count, int experts)
{
    const bool mixed = hasMappedGPUCPUIntermediateShardTopology(plan) &&
        plan.authority_execution == MoEOverlayAuthorityExecutionKind::HostResident;
    if (!device.is_gpu() || world_rank < 0 || first_layer < 0 || layer_count <= 0 || experts <= 0 ||
        first_layer > std::numeric_limits<int>::max() - layer_count ||
        !plan.usesExpertOverlayAuthority() || (!mixed && plan.routed_tiers.size() != 1) ||
        plan.continuation_domain != plan.routed_tiers.front().domain ||
        plan.shared_expert_domain != plan.continuation_domain || plan.first_model_layer != first_layer)
        throw std::invalid_argument("GPU F-shard graph requires its original routed/shared/continuation source and exact stage scope");
    const auto domain = std::find_if(plan.domains.begin(), plan.domains.end(),
        [&](const auto &candidate) { return candidate.name == plan.continuation_domain; });
    if (domain == plan.domains.end() || domain->scope != ExecutionDomainScope::RANK_LOCAL ||
        domain->routed_compute_policy != RoutedExpertComputePolicy::TensorSharded ||
        domain->routed_phase_policy != RoutedExpertPhasePolicy::Uniform ||
        domain->routed_decode_assignment_policy != RoutedExpertAssignmentPolicy::StaticOwner ||
        domain->routed_prefill_assignment_policy != RoutedExpertAssignmentPolicy::StaticOwner ||
        domain->participants.empty() || domain->participants.size() != std::size_t(context.degree()) ||
        context.devices().size() != domain->participants.size() || member < 0 ||
        std::size_t(member) >= domain->participants.size())
        throw std::invalid_argument("Static GPU F-shard graph changed original rank-local domain, phase policy or positive membership");
    if (domain->owner_rank != world_rank ||
        (domain->backend != CollectiveBackendType::AUTO && domain->backend != context.backend()))
        throw std::invalid_argument("Static GPU F-shard graph changed authored original owner rank or collective backend");
    const bool identity = context.degree() == 1 && context.backend() == CollectiveBackendType::HOST;
    if (!identity && context.backend() != (device.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL))
        throw std::invalid_argument("Static GPU F-shard graph requires its original homogeneous native backend; HOST is only a singleton identity");
    const MoEExpertOwnerParticipant *selected = nullptr;
    for (const auto &candidate : owners.participants())
        if (candidate.domain_name == domain->name && candidate.domain_participant_index == member)
        {
            if (selected) throw std::invalid_argument("Static GPU F-shard graph has ambiguous original member");
            selected = &candidate;
        }
    if (!selected || selected->device != device || !selected->world_rank_known || selected->world_rank != world_rank)
        throw std::invalid_argument("Static GPU F-shard graph changed original device or process ownership");
    for (int layer = first_layer; layer < first_layer + layer_count; ++layer)
    for (int expert = 0; expert < experts; ++expert)
    {
        const auto *row = owners.placementFor(layer, expert);
        if (!row || !row->shard_cohort || !row->resident ||
            row->shard_cohort->source().layer_idx != layer)
            throw std::invalid_argument("GPU F-shard graph omits an original main/predictor logical expert cohort at layer " + std::to_string(layer));
        if (mixed && row->tier_idx == 1 && row->domain_name == plan.routed_tiers.back().domain)
        {
            const auto &cpu = *std::find_if(plan.domains.begin(), plan.domains.end(),
                [&](const auto &candidate) { return candidate.name == row->domain_name; });
            if (row->shard_cohort->participants().size() != cpu.participants.size())
                throw std::invalid_argument("Mapped CPU expert cohort lost its original cardinality");
            for (size_t index = 0; index < cpu.participants.size(); ++index)
            {
                const auto &original = row->shard_cohort->participants()[index];
                const auto *owner = owners.participantForId(original.participant_id);
                if (!owner || !owner->world_rank_known || owner->world_rank != original.world_rank ||
                    owner->address != original.address || original.address != cpu.participants[index] ||
                    owner->tier_idx != 1 || owner->domain_name != cpu.name ||
                    owner->domain_participant_index != static_cast<int>(index))
                    throw std::invalid_argument("Mapped CPU expert cohort changed original process or native fragment order");
            }
            continue;
        }
        if (row->tier_idx != 0 || row->domain_name != domain->name ||
            row->shard_cohort->participants().size() != domain->participants.size())
            throw std::invalid_argument("GPU F-shard graph received a foreign initial expert cohort");
        for (std::size_t index = 0; index < domain->participants.size(); ++index)
        {
            const auto &original = row->shard_cohort->participants()[index];
            const auto *owner = owners.participantForId(original.participant_id);
            if (!owner || original.world_rank != world_rank || !original.address.isGPU() ||
                original.address.device_type != selected->address.device_type ||
                original.address != context.devices()[index] || original.address != domain->participants[index] ||
                owner->domain_name != domain->name || owner->domain_participant_index != int(index) ||
                owner->address != original.address || !owner->world_rank_known || owner->world_rank != world_rank ||
                (index == std::size_t(member) && original.participant_id != selected->participant_id))
                throw std::invalid_argument("Static GPU F-shard graph changed original global/cohort member or native reduction order");
        }
    }
    return *selected;
}

/** @brief Preserve the existing one-tier graph-admission negative controls. */
inline const MoEExpertOwnerParticipant &requireOriginalStaticGPUIntermediateShardGraph(
    const MoERoutedExpertPlacementPlan &plan, const MoEExpertOwnerMap &owners,
    const ILocalTPContext &context, DeviceId device, int member, int world_rank,
    int first_layer, int layer_count, int experts)
{
    if (plan.routed_tiers.size() != 1)
        throw std::invalid_argument("Static GPU F-shard graph requires a single routed cohort");
    return requireOriginalGPUIntermediateShardGraph(
        plan, owners, context, device, member, world_rank, first_layer, layer_count, experts);
}
}
