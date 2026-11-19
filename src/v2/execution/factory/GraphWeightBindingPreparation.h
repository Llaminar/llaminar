/**
 * @file GraphWeightBindingPreparation.h
 * @brief Normalize original expert sources once at the graph preparation boundary.
 *
 * Materialization owns native compact expert parents; graph preparation consumes
 * per-expert matrix views. This setup-only handoff preserves the same physical
 * source, participant identity and memory authority across the dense and explicit
 * expert passes. Already projected bindings retain their views after source
 * retirement, including when a graph reuses an existing prepared bank.
 */
#pragma once

#include "loaders/WeightPlan.h"
#include "tensors/TensorClasses.h"

namespace llaminar2
{
    /**
     * @brief Publish matrix views for original compact expert parents, if present.
     * @param source Frozen materialization result for this graph participant.
     * @return The original bindings or their zero-copy per-expert matrix views.
     * @throws std::exception If source identity, ownership or geometry is invalid.
     *
     * Dense bindings and already committed CPU/GPU matrices remain unchanged.
     * The existing strict projection API authenticates every source parent; this
     * helper never reloads, repacks or recreates a previously projected source.
     */
    inline FrozenModelWeightSet projectGraphExpertSourceBindings(FrozenModelWeightSet source)
    {
        source.validateForGraph();
        for (const auto &binding : source.bindings())
            if (binding.projection_ownership && binding.tensor && binding.tensor->shape().size() == 3)
                return source.projectedExpertMatrices();
        return source;
    }
}
