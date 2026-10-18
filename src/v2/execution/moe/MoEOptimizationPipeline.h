/**
 * @file MoEOptimizationPipeline.h
 * @brief Checked composition of passive, independent MoE stage publications.
 *
 * Only work counts are additive. Demand windows, decision receipts and epochs
 * remain in their stage namespaces, and terminal histories retain the same
 * structure so equal transaction numbers cannot collide or be deduplicated.
 */
#pragma once
#include "MoEOptimizationStatus.h"
#include "MoEOptimizationMovementTopology.h"

namespace llaminar2
{
    /**
     * @brief Compose all stage status observations without advancing their owners.
     * @param stages Complete ordered single-authority observations.
     * @return Checked additive summary and immutable per-stage detail.
     * @throws std::invalid_argument for malformed scope or authority state.
     * @throws std::overflow_error if any additive counter cannot be represented.
     */
    MoEOptimizationStatus composeMoEOptimizationStatus(
        MoEOptimizationStages<MoEOptimizationStatus> stages);

    /**
     * @brief Mark already-retired publication owners terminal without losing their scopes.
     * @param status Final observation after the caller has joined every stage owner.
     * @return Drained single-authority or fully scoped pipeline observation.
     *
     * This publishes a lifecycle fact; it never drains or synchronizes devices.
     * A failed owner remains failed and cannot be relabeled as successful retirement.
     */
    MoEOptimizationStatus drainedMoEOptimizationStatus(MoEOptimizationStatus status);

    /**
     * @brief Bind complete stage ledgers to their exact layer/participant namespaces.
     * @param stages Complete ordered histories; truncation remains explicit.
     * @return Hierarchical ledger with no synthetic global transaction numbers.
     * @throws std::invalid_argument for foreign layer or participant evidence.
     */
    MoEOptimizationMovementLedger composeMoEOptimizationMovementLedger(
        MoEOptimizationStages<MoEOptimizationMovementLedger> stages);

    /**
     * @brief Compose frozen per-stage movement geometry, without inferring activity.
     * @param stages Complete ordered frozen-plan projections.
     * @return Union of available axes plus the individual authority namespaces.
     * @throws std::invalid_argument for invalid or missing stage geometry.
     */
    MoEOptimizationMovementTopology composeMoEOptimizationMovementTopology(
        MoEOptimizationStages<MoEOptimizationMovementTopology> stages);
}
