/**
 * @file Qwen4ExpAllGPUQ6MTPParityCases.h
 * @brief Exact moving-MTP candidates on the original CUDA2/ROCm4 GPU tiers with RAM PLE.
 *
 * This derives the authored Q6 hardware row. CUDA keeps its eight-expert quota;
 * the original four-ROCm tier covers all remaining experts without a routed CPU
 * tier. RAM ngrams remain a host source, not a CPU expert participant. Original
 * PMA/source admission prices every weight, migration, graph and snapshot byte;
 * these declarations confer no capacity, numerical or native qualification.
 */
#pragma once

#include "Qwen4ExpGPUAdaptiveParityCases.h"

namespace llaminar2::test::parity::qwen4exp
{
/**
 * @brief Project the existing six-GPU workload into complete GPU expert residency.
 * @return Original model/math and CUDA2/ROCm4 domains, ordinary RAM PLE and no read-ahead.
 * @throws std::logic_error If the original Q6 tier/participant boundary changes.
 * @note Only the CPU routed capacity domain is removed. CPU RAM source ownership
 *       is still resolved independently by the original lookup/PMA machinery.
 */
inline ModelParityDefinition qwen4ExpAllGPUQ6MTPDefinition()
{
    auto definition = qwen4ExpHardwareTargetDefinition(Qwen4ExpHardwareTarget::CUDA2ROCm4);
    auto plan = std::make_shared<MoERoutedExpertPlacementPlan>(*definition.topology.expert_overlay_plan);
    if (plan->domains.size() != 3 || plan->routed_tiers.size() != 3 ||
        plan->domains[0].participants.size() != 2 || plan->domains[1].participants.size() != 4 ||
        plan->domains[2].participants.size() != 1 || !plan->domains[2].participants.front().isCPU() ||
        plan->routed_tiers[0].domain != plan->domains[0].name ||
        plan->routed_tiers[1].domain != plan->domains[1].name ||
        plan->routed_tiers[2].domain != plan->domains[2].name ||
        plan->routed_tiers[0].max_experts_per_layer != 8)
        throw std::logic_error("All-GPU Q6 lost its original CUDA2/ROCm4/CPU tier boundary");
    plan->domains.pop_back();
    plan->routed_tiers.pop_back();
    plan->routed_tiers.back().max_experts_per_layer = 0;
    plan->routed_tiers.back().memory_budget_bytes = 0;
    plan->routed_tiers.back().fallback = true;
    std::erase_if(definition.topology.participants, [](const auto& participant) {
        return participant.address.isCPU();
    });
    if (definition.topology.participants.size() != 6 ||
        resolveMoEOverlayAuthorityExecutionKind(*plan) != MoEOverlayAuthorityExecutionKind::DeviceResident ||
        !validateMoERoutedExpertPlacementPlan(*plan).ok())
        throw std::logic_error("All-GPU Q6 requires its exact original six GPU owners");
    definition.topology.expert_overlay_plan = std::move(plan);
    definition.topology.test_id = "CUDA0_CUDA1_TP2_ExpertOverlay_GpuExperts8_ROCm0123Followers_AllGPUExperts";
    definition.features.ngram_storage = {{.storage = ImmutableRowStoragePreference::RAM}};
    definition.features.mtp = ModelParityAxisProfile::Disabled;
    definition.features.dynamic_speedup_witness = ModelParityDynamicSpeedupWitness::Disabled;
    definition.model.shared_predictor.reset();
    definition.model.maximum_mtp_draft_depth = 0;
    return definition;
}

/**
 * @return Exactly fixed1/fixed3/dynamic native candidates at original DynamicOrdinal placement.
 * @throws std::logic_error If the original expander loses its unique RAM moving seed.
 * @note Retained3 bounds this functional slice. All original movement, exact-byte,
 *       source, graph, reset and retirement checks remain active; performance and
 *       frozen-history numerical qualification are separate obligations.
 */
inline const std::vector<ModelParityCase>& qwen4ExpAllGPUQ6MTPNativeExactCases()
{
    static const auto cases = [] {
        const ModelParityExpertOverlayPolicy moving{
            .owner_order = RoutedExpertOwnerOrder::Ordinal, .movement = ModelParityExpertMovement::Dynamic};
        std::vector<ModelParityCase> seeds;
        for (auto cell : expandModelParityDefinition(qwen4ExpAllGPUQ6MTPDefinition()))
            if (cell.expert_overlay == std::optional{moving} && cell.mtp == ModelParityMTP::Off)
                seeds.push_back(std::move(cell));
        if (seeds.size() != 1)
            throw std::logic_error("All-GPU Q6 moving RAM exact slice lost its unique original seed");
        std::vector<ModelParityCase> result;
        for (const auto mtp : {ModelParityMTP::Depth1, ModelParityMTP::Depth3, ModelParityMTP::DynamicDepth}) {
            auto cell = seeds.front();
            cell.inventory_role = ModelParityInventoryRole::NumericalDiagnostic;
            cell.model.shared_predictor = SharedPredictorGGUFSource{std::string(kPredictorModelPath)};
            cell.model.maximum_mtp_draft_depth = 3;
            cell.retained_mtp_draft_capacity = 3;
            cell.mtp = mtp;
            cell.topology.test_id += "_R3";
            result.push_back(std::move(cell));
        }
        return result;
    }();
    return cases;
}
} // namespace llaminar2::test::parity::qwen4exp
