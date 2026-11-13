/**
 * @file PreparedPipelineGraphMaterialization.h
 * @brief Close one rank-owned prepared source after its complete global pipeline family.
 *
 * Nested native TP stages retain shared source bytes until the enclosing rank
 * proves every actual stage graph is materialized. This boundary publishes no
 * completion on a failed capture or ownership check. It owns no model, graph,
 * stream or memory ledger; the original WeightManager performs typed release.
 */
#pragma once
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "loaders/WeightManager.h"
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Materialize and authenticate all local stages before releasing shared sources.
     * @param runner Actual original-rank global pipeline graph owner.
     * @param weights Same model-owned manager that prepared every local TP/MTP operand.
     * @param plan Complete admitted serving graph family, including retained MTP variants.
     * @return False if actual family materialization failed, without publishing completion.
     * @throws std::logic_error For incomplete source preparation or failed ownership proof.
     * @throws std::exception Propagates capture, ownership and typed source-release failures.
     */
    inline bool materializePreparedPipelineGraphFamily(IInferenceRunner &runner,
        WeightManager &weights, const ServingGraphFamilyMaterializationPlan &plan)
    {
        const auto &gates = weights.lifecycleGates();
        if (!gates.materialization_complete || !gates.device_preparation_complete)
            throw std::logic_error("Global pipeline graph completion requires every original rank-owned prepared operand");
        if (!runner.materializeServingGraphFamilyWithoutLaunch(plan))
            return false;
        runner.requireMoEPipelineRuntimeOwnership(
            IInferenceRunner::MoEPipelineOwnershipValidation::ServingMaterialized);
        // The complete parent, including terminal predictor bindings, is now
        // retained. RequiredForCPUExecution operands remain governed by their
        // original host policy inside WeightManager; graph success never
        // grants permission to discard CPU execution storage.
        weights.markGraphMaterializationComplete();
        weights.releaseAllHostWeightData();
        return true;
    }
}
