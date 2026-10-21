/**
 * @file Qwen4ExpROCm4OverlayPolicyParityCases.h
 * @brief Exact dense/routed policy controls beside the unchanged four-ROCm QFN cell.
 *
 * All participants, source roles, RAM PLE, numerical limits and phase evidence
 * derive from the original full-residency definition. These three additional
 * declarations complete its two-by-two policy comparison without changing
 * automatic defaults, existing case IDs or assigning admission capacity.
 */
#pragma once
#include "Qwen4ExpFullROCmTP4ParityCases.h"
#include <array>
#include <utility>

namespace llaminar2::test::parity::qwen4exp
{
/**
 * @brief Derive one explicit policy comparison from the original TP4 definition.
 * @param dense TensorParallel or PrefillTensorParallelDecodeReplicated.
 * @param routed GateUpOwnedDownColumns or Apportioned, never expert replication.
 * @return Independent immutable source/topology blueprint, with original A1 ID retained for its pair.
 * @throws std::invalid_argument For a policy outside this explicit comparison.
 * @note Replicated dense decode requests its real additional source/cache BOM;
 *       sole production PMA admission may reject it. No other placement is substituted.
 */
inline ModelParityDefinition qwen4ExpROCm4OverlayPolicyDefinition(
    DenseParallelPolicy dense, RoutedExpertComputePolicy routed)
{
    if ((dense != DenseParallelPolicy::TensorParallel &&
         dense != DenseParallelPolicy::PrefillTensorParallelDecodeReplicated) ||
        (routed != RoutedExpertComputePolicy::GateUpOwnedDownColumns &&
         routed != RoutedExpertComputePolicy::Apportioned))
        throw std::invalid_argument("QFN ROCm4 comparison requires explicit supported dense/routed policies");
    auto definition = qwen4ExpFullROCmTP4Definition();
    if (dense == DenseParallelPolicy::TensorParallel &&
        routed == RoutedExpertComputePolicy::GateUpOwnedDownColumns)
        return definition;
    auto plan = std::make_shared<MoERoutedExpertPlacementPlan>(*definition.topology.expert_overlay_plan);
    plan->continuation_domain_spec.setDensePolicy(dense);
    plan->domains.front().routed_compute_policy = routed;
    // Dense and routed spellings refer to the SAME original hardware domain.
    // Reproject the full declaration rather than leaving conflicting policy metadata.
    plan->dense_domains = {plan->domains.front().toExecutionDomainDefinition()};
    const char *dense_name = dense == DenseParallelPolicy::TensorParallel
        ? "TensorParallel" : "PrefillTensorParallelDecodeReplicated";
    const char *routed_name = routed == RoutedExpertComputePolicy::Apportioned
        ? "Apportioned" : "GateUpOwnedDownColumns";
    definition.topology.test_id = std::string("ROCm0123_") + dense_name + "_" + routed_name;
    definition.topology.expert_overlay_plan = std::move(plan);
    if (!validateMoERoutedExpertPlacementPlan(*definition.topology.expert_overlay_plan).ok())
        throw std::invalid_argument("Invalid QFN ROCm4 dense/routed comparison declaration");
    return definition;
}

/**
 * @return Exactly three additional StaticOrdinal/MTP-off/RAM cells through the original expander.
 * @note Existing A1 remains in its original family. No automatic storage/MTP/precision axes are added.
 */
inline const std::vector<ModelParityCase> &qwen4ExpROCm4OverlayPolicyProductionCases()
{
    static const auto cells = [] {
        constexpr std::array policies{
            std::pair{DenseParallelPolicy::TensorParallel, RoutedExpertComputePolicy::Apportioned},
            std::pair{DenseParallelPolicy::PrefillTensorParallelDecodeReplicated,
                      RoutedExpertComputePolicy::GateUpOwnedDownColumns},
            std::pair{DenseParallelPolicy::PrefillTensorParallelDecodeReplicated,
                      RoutedExpertComputePolicy::Apportioned}};
        std::vector<ModelParityCase> result;
        for (const auto &[dense, routed] : policies)
            for (auto &cell : expandModelParityDefinition(qwen4ExpROCm4OverlayPolicyDefinition(dense, routed)))
                if (cell.mtp == ModelParityMTP::Off &&
                    cell.expert_overlay == std::optional{ModelParityExpertOverlayPolicy{}})
                    result.push_back(std::move(cell));
        if (result.size() != policies.size())
            throw std::logic_error("QFN ROCm4 policy comparison must contain exactly three additional cells");
        return result;
    }();
    return cells;
}
} // namespace llaminar2::test::parity::qwen4exp
