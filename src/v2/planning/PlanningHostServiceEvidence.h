/**
 * @file PlanningHostServiceEvidence.h
 * @brief Bounded discovery-rank CPU service evidence for metadata-only admission.
 *
 * These declarations contain source identities, never live workers, pointers,
 * memory reservations or caller-supplied byte limits. Candidate membership maps
 * discovery evidence into execution ranks; the selected process must still
 * authenticate its original local program before materialization.
 */
#pragma once

#include "execution/local_execution/graph/RetainedHostServiceDeclarations.h"
#include <cstdint>
#include <map>
#include <span>
#include <vector>

namespace llaminar2 {
class AutomaticPlanningPreparation;
class ExecutionRankMembership;

/** @brief Immutable per-discovery-rank classes, without a second admission ledger. */
using PlanningHostServiceBudgets = std::map<int, RetainedHostServiceBudgetProjection>;

/**
 * @brief Encode at most the two authored service roles using fixed-size identities.
 * @param classes Original rank declarations, without native owners or numeric limits.
 * @return One versioned frame with exact role and digest extents.
 * @throws std::invalid_argument for duplicate/unknown roles or malformed identities.
 */
std::vector<std::uint8_t> encodePlanningHostServiceClasses(
    std::span<const RetainedHostServiceClassProjection> classes);

/**
 * @brief Authenticate the exact bounded wire frame before any class selection.
 * @param bytes Complete original rank receipt from the planning transport.
 * @return Validated role identities; this does not qualify their native classes.
 * @throws std::invalid_argument for trailing, truncated or incompatible evidence.
 */
std::vector<RetainedHostServiceClassProjection> decodePlanningHostServiceClasses(
    std::span<const std::uint8_t> bytes);

/**
 * @brief Gather source-bound original classes through failure-atomic startup exchange.
 * @param context Original discovery communicator, model metadata and inventory.
 * @return Root-only complete rank map; followers return no remote native capability.
 */
PlanningHostServiceBudgets collectPlanningHostServiceBudgets(
    const AutomaticPlanningPreparation& context);

/**
 * @brief Select the original discovery receipt for one compiled execution rank.
 * @param budgets Complete available original discovery evidence.
 * @param membership Compiler-owned discovery-to-execution mapping.
 * @param executionRank Selected rank in the compacted execution namespace.
 * @return That exact projection, or an explicitly empty projection when no evidence
 * was supplied. Attempting to declare any service from an empty projection fails.
 * @throws std::out_of_range for an invalid execution-rank coordinate.
 */
RetainedHostServiceBudgetProjection projectPlanningHostServiceBudget(
    const PlanningHostServiceBudgets& budgets,
    const ExecutionRankMembership& membership, int executionRank);

/**
 * @brief Retain the single declaration session already attached to rank-local inputs.
 * @param inputs Exact participant inputs returned by the fixed-owner planner.
 * @param rank Physical execution rank owning every supplied input.
 * @return Same declaration owner, or absence when this rank has no inputs.
 * @throws std::logic_error for missing, mixed or foreign declarations.
 */
std::shared_ptr<const RetainedHostServiceDeclarations> retainedHostServicesFromInputs(
    std::span<const struct DevicePlanConfig> inputs, int rank);
}
