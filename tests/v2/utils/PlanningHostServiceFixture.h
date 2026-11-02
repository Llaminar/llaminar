/**
 * @file PlanningHostServiceFixture.h
 * @brief Closed device-free CPU service evidence for canonical planning replays.
 *
 * Recorded opaque values are independent test inputs to production BOM logic.
 * No source discovery, native worker, allocator or installed certificate is
 * substituted; these helpers cannot qualify real libc/TLS backing.
 */
#pragma once
#include "../fixtures/execution/RetainedHostServiceReplayAccess.h"
#include "planning/PlanningHostServiceEvidence.h"

namespace llaminar2::test {
/** @return Both authored roles with recorded opaque demand and actual stack geometry. */
inline RetainedHostServiceBudgetProjection planningHostServiceProjection(std::size_t opaque = 2u<<20) {
    return RetainedHostServiceReplayAccess::projection({
        RetainedHostServiceReplayAccess::program(RetainedHostServiceRole::ImmutableLookupIO, opaque),
        RetainedHostServiceReplayAccess::program(RetainedHostServiceRole::ExecutorParentTicket, opaque)});
}

/** @return One independently labeled recorded projection for every discovery rank. */
inline PlanningHostServiceBudgets planningHostServiceBudgets(int rankCount) {
    if (rankCount < 0) throw std::invalid_argument("Recorded service fixture requires a nonnegative rank count");
    PlanningHostServiceBudgets result;
    for (int rank = 0; rank < rankCount; ++rank)
        result.emplace(rank, planningHostServiceProjection());
    return result;
}

/**
 * @brief Freeze explicitly authored fixture instances through the production builder.
 * @param rank Original physical rank owning every supplied instance.
 * @param instances Exact fixture roles; this helper never infers topology or counts.
 * @return One shared declaration for planning and local wire reconstruction.
 */
inline std::shared_ptr<const RetainedHostServiceDeclarations> planningHostServiceDeclarations(
    int rank, std::span<const RetainedHostServiceInstance> instances) {
    RetainedHostServiceDeclarations::Builder builder(rank);
    const auto projection = planningHostServiceProjection();
    for (const auto& instance : instances) builder.addProjected(instance, projection);
    return std::move(builder).freeze();
}
}
