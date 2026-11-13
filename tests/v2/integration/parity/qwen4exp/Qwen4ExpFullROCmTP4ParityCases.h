/**
 * @file Qwen4ExpFullROCmTP4ParityCases.h
 * @brief One GPU-only expert-residency four-ROCm main-model diagnostic.
 *
 * These declarations retain the original GGUF/reference, complete 48-layer raw
 * evidence, prefill+two teacher-forced decode observations and prefix/path
 * gates. RAM PLE is independent of routed compute: no CPU expert domain or
 * eight-expert GPU limit is permitted. The sole final GPU tier must cover all
 * original experts or production admission fails; no capacity is granted here.
 * TP4 uses the current homogeneous projection policy explicitly as a matched
 * topology control. RAM PLE remains host-owned; full GPU residency describes
 * all routed experts, not the ngram table. This cell grants no serving,
 * numerical, performance, Auto, E2E or MTP qualification merely by declaration.
 * The local-NUMA sibling keeps MPI ordinal independent of physical NUMA identity
 * and requests full15 MTP with bounded ConsumerLocal caches on a RAM source.
 */
#pragma once
#include "Qwen4ExpProductionParityCases.h"
#include "Qwen4ExpGPUAdaptiveParityCases.h"
#include <span>
#include <set>

namespace llaminar2::test::parity::qwen4exp
{
/**
 * @brief Declare all four physical ROCm participants with complete GPU expert ownership.
 * @return Exact homogeneous rank-local TP4 topology; physical capacity remains production-PMA owned.
 * @throws std::invalid_argument For an invalid overlay declaration.
 */
inline ModelParityTopologyDefinition qwen4ExpFullROCmTP4Topology()
{
    ModelParityTopologyDefinition topology;
    topology.mpi_ranks = 1;
    topology.collective = Collective::RCCL;
    for (int ordinal = 0; ordinal < 4; ++ordinal)
    {
        topology.participants.push_back({GlobalDeviceAddress::rocm(ordinal), 0});
    }
    topology.test_id = "ROCm0123_TP4_AllGPUExperts";
    topology.kind = ModelParityTopologyKind::RankLocalTensorParallel;
    RoutedExpertDomain domain;
    domain.name = "qfn_rocm_full_tp";
    domain.scope = ExecutionDomainScope::RANK_LOCAL;
    domain.backend = CollectiveBackendType::RCCL;
    domain.owner_rank = 0;
    for (const auto &participant : topology.participants)
    {
        domain.participants.push_back(participant.address);
    }
    domain.weights.assign(4, 0.25f);
    domain.routed_compute_policy = RoutedExpertComputePolicy::GateUpOwnedDownColumns;
    domain.routed_phase_policy = RoutedExpertPhasePolicy::Uniform;
    domain.routed_decode_assignment_policy = RoutedExpertAssignmentPolicy::StaticOwner;
    domain.routed_prefill_assignment_policy = RoutedExpertAssignmentPolicy::StaticOwner;
    auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
    plan->enabled = true;
    plan->topology = RoutedExpertPlacementTopology::SingleDomain;
    plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
    plan->owner_order = RoutedExpertOwnerOrder::Ordinal;
    plan->continuation_domain = plan->base_model_domain = plan->shared_expert_domain = domain.name;
    plan->continuation_domain_spec.domain = domain.name;
    plan->continuation_domain_spec.logical_root_participant = 0;
    plan->continuation_domain_spec.setDensePolicy(DenseParallelPolicy::TensorParallel);
    plan->continuation_domain_spec.hidden_layout = MoEContinuationActivationLayout::ReplicatedHidden;
    plan->dense_domains = {domain.toExecutionDomainDefinition()};
    plan->domains = {domain};
    // Zero is the canonical uncapped final coverage request, not zero slots.
    // No lower CPU tier exists to accept an incomplete original expert set.
    plan->routed_tiers = {{.name = "all_gpu", .domain = domain.name, .priority = 0,
        .max_experts_per_layer = 0, .memory_budget_bytes = 0, .fallback = true}};
    if (!validateMoERoutedExpertPlacementPlan(*plan).ok())
        throw std::invalid_argument("Invalid full-residency ROCm TP4 expert topology");
    topology.expert_overlay_plan = std::move(plan);
    return topology;
}

/**
 * @brief Retain the canonical main numerical/source contract and request RAM PLE.
 * @return Unchanged ordinary precisions, references, snapshots and numerical limits.
 */
inline ModelParityDefinition qwen4ExpFullROCmTP4Definition()
{
    auto definition = qwen4ExpProductionDefinition(Qwen4ExpParityTopology::CPU);
    definition.topology = qwen4ExpFullROCmTP4Topology();
    // The current fixed-history corpus authenticates this inactive paired source.
    // Its declaration does not enable MTP or reserve predictor graph capacity.
    definition.model.shared_predictor = SharedPredictorGGUFSource{std::string(kPredictorModelPath)};
    definition.features.ngram_storage = {{.storage = ImmutableRowStoragePreference::RAM}};
    return definition;
}

/**
 * @return Exactly one TP4 StaticOrdinal/MTP-off case, through the existing expander.
 * @note All broader policy axes remain future qualification; old families are unchanged.
 */
inline const std::vector<ModelParityCase> &qwen4ExpFullROCmTP4ProductionCases()
{
    static const auto cases = [] {
        std::vector<ModelParityCase> result;
        for (auto &cell : expandModelParityDefinition(qwen4ExpFullROCmTP4Definition()))
        {
            if (cell.mtp == ModelParityMTP::Off &&
                cell.expert_overlay == std::optional{ModelParityExpertOverlayPolicy{}})
            {
                result.push_back(std::move(cell));
            }
        }
        if (result.size() != 1)
            throw std::logic_error("Full ROCm diagnostic inventory must contain exactly one TP4 case");
        return result;
    }();
    return cases;
}
/**
 * @brief Rebind the existing all-GPU policy to one ordered physical NUMA-local group.
 * @param members Ordered homogeneous ROCm participants with explicit matching NUMA IDs.
 * @return One actual MPI rank owning every GPU and every routed expert; RAM is independent storage.
 * @throws std::invalid_argument For fewer than two members, mixed/unknown locality or duplicate GPUs.
 * @details Rank0 is communicator identity, not a NUMA ID. The launcher must bind
 * its sole process to this observed socket; production ownership checks verify it.
 * Both dense and routed declarations share the same ordered physical endpoints.
 */
inline ModelParityTopologyDefinition qwen4ExpLocalNUMAAllGPUContinuationTopology(
    std::span<const GlobalDeviceAddress> members)
{
    if(members.size()<2)throw std::invalid_argument("All-GPU dense TP requires at least two ROCm participants");
    const auto& first=members.front();std::set<int> ordinals;
    for(const auto& member:members)
        if(!member.isROCm() || !member.hasValidNuma() || member.numa_node!=first.numa_node ||
           member.hostname!=first.hostname || member.device_ordinal<0 ||
           !ordinals.insert(member.device_ordinal).second)
            throw std::invalid_argument("All-GPU continuation requires ordered unique ROCm devices on one explicit NUMA node");
    auto topology=qwen4ExpFullROCmTP4Topology();
    topology.participants.clear();topology.test_id.clear();
    for(const auto& member:members) {
        topology.participants.push_back({member,0});
        if(!topology.test_id.empty())topology.test_id+='_';
        topology.test_id+="ROCm"+std::to_string(member.device_ordinal);
    }
    topology.test_id+="_NUMA"+std::to_string(first.numa_node)+"_AllGPUExperts_DenseTP_GateUpOwnedDownColumns";
    auto plan=std::make_shared<MoERoutedExpertPlacementPlan>(*topology.expert_overlay_plan);
    auto& domain=plan->domains.front();
    domain.participants.assign(members.begin(),members.end());
    domain.weights.assign(members.size(),1.0f/static_cast<float>(members.size()));
    plan->dense_domains={domain.toExecutionDomainDefinition()};
    const auto validation=validateMoERoutedExpertPlacementPlan(*plan);
    if(!validation.ok())throw std::invalid_argument("Invalid all-GPU continuation: "+validation.errors.front());
    topology.expert_overlay_plan=std::move(plan);
    return topology;
}

/**
 * @return Original all-expert GPU policy on ROCm1..4/NUMA1 with RAM source, 256MiB local caches and full15 MTP.
 * @note A single final GPU tier owns the entire512 expert set or PMA fails.
 * No CPU compute participant or lower tier can absorb an incomplete GPU admission.
 */
inline ModelParityDefinition qwen4ExpLocalNUMATP4AllGPUDefinition()
{
    auto definition=qwen4ExpFullROCmTP4Definition();
    const std::array members{GlobalDeviceAddress::rocm(1,1),GlobalDeviceAddress::rocm(2,1),
        GlobalDeviceAddress::rocm(3,1),GlobalDeviceAddress::rocm(4,1)};
    definition.topology=qwen4ExpLocalNUMAAllGPUContinuationTopology(members);
    definition.features.ngram_storage=qwen4ExpConsumerLocalTP4MovementDefinition(
        AdaptiveImmutableRowBacking::RAM).features.ngram_storage;
    definition.features.mtp=ModelParityAxisProfile::Standard;
    definition.features.dynamic_speedup_witness=ModelParityDynamicSpeedupWitness::Disabled;
    definition.model.maximum_mtp_draft_depth=kModelParityRequiredMaximumMTPDepth;
    // The canonical expander validates both declared movement policies before
    // selecting StaticOrdinal cells. Retain its complete full15 transaction
    // envelope even though this family's static runtime never moves experts.
    for(auto* movement:{&definition.dynamic_rebalance.economic_movement,
                        &definition.dynamic_rebalance.economic_movement_and_observed_speedup}) {
        movement->device_min_maintenance_period_tokens=kModelParityRequiredMaximumMTPDepth+1;
        movement->device_initial_maintenance_period_tokens=kModelParityRequiredMaximumMTPDepth+1;
    }
    return definition;
}

/** @return Four StaticOrdinal cells: Off/1/3/Dynamic, retaining full15 for active MTP only. */
inline const std::vector<ModelParityCase>& qwen4ExpLocalNUMATP4AllGPUCases()
{
    static const auto cases=[] {
        std::vector<ModelParityCase> result;
        for(auto cell:expandModelParityDefinition(qwen4ExpLocalNUMATP4AllGPUDefinition()))
            if(cell.expert_overlay==std::optional{ModelParityExpertOverlayPolicy{}} &&
               (cell.mtp==ModelParityMTP::Off || cell.mtp==ModelParityMTP::Depth1 ||
                cell.mtp==ModelParityMTP::Depth3 || cell.mtp==ModelParityMTP::DynamicDepth)) {
                cell.inventory_role=ModelParityInventoryRole::NumericalDiagnostic;
                cell.retained_mtp_draft_capacity=cell.mtp==ModelParityMTP::Off?0:kModelParityRequiredMaximumMTPDepth;
                result.push_back(std::move(cell));
            }
        if(result.size()!=4)throw std::logic_error("Local NUMA all-GPU TP4 lost its original MTP policy axes");
        return result;
    }();return cases;
}

/** @return Active native-exact proofs of the identical all-GPU production/export policies. */
inline std::vector<ModelParityCase> qwen4ExpLocalNUMATP4AllGPUExactCases()
{
    auto result=qwen4ExpLocalNUMATP4AllGPUCases();
    std::erase_if(result,[](const auto& cell){return cell.mtp==ModelParityMTP::Off;});
    return result;
}
} // namespace llaminar2::test::parity::qwen4exp
