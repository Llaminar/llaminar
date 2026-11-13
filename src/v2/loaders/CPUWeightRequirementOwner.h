/**
 * @file CPUWeightRequirementOwner.h
 * @brief Preserve original routed source purpose inside a mixed CPU graph plan.
 *
 * Graph composition retains a continuation-wide owner while appending original
 * participant source declarations. This pure projection preserves their routed
 * identity without capacity arithmetic, allocation, readiness or another ledger.
 */
#pragma once
#include "WeightPlan.h"
namespace llaminar2 {
/**
 * @brief Resolve one original CPU source's exact persistent physical owner.
 * @param requirement Original source/participant declaration retained by the plan.
 * @param declared_owner Ordinary plan owner, unchanged for non-overlay sources.
 * @return RoutedExpertWeights only for an authenticated overlay expert source.
 * @throws std::invalid_argument For a routed declaration with foreign source purpose.
 */
inline PhysicalMemoryOwner resolveCPUWeightRequirementOwner(
    const WeightRequirement &requirement, PhysicalMemoryOwner declared_owner)
{
    const auto role = requirement.role == WeightRole::Other ? inferWeightRole(requirement.canonical_name) : requirement.role;
    if (!requirement.overlay_domain.empty() && isRoutedExpertRole(role)) {
        // A graph-local mixed plan retains its continuation owner, while
        // these original participant sources belong to the separate routed
        // BOM. Validate the source purpose before choosing that owner; a
        // CPU role alone never changes an ordinary plan's declaration.
        const auto &source_name = requirement.source_name.empty() ? requirement.canonical_name : requirement.source_name;
        if (inferWeightRole(source_name) != role || inferWeightRole(requirement.canonical_name) != role ||
            requirement.derivation != WeightDerivationKind::ExpertSlice ||
            requirement.overlay_participant_index < 0 || requirement.overlay_participant_world_rank < -1 ||
            requirement.slice.expert_ids.empty())
            throw std::invalid_argument("CPU routed source requires its original overlay participant/expert identity");
        return PhysicalMemoryOwner::RoutedExpertWeights;
    }
    return declared_owner;
}
}
