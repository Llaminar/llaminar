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
 */
#pragma once
#include "Qwen4ExpProductionParityCases.h"

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
} // namespace llaminar2::test::parity::qwen4exp
