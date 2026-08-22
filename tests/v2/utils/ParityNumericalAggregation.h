/**
 * @file ParityNumericalAggregation.h
 * @brief Typed rules for aggregating heterogeneous MoE parity checkpoints.
 *
 * Every checkpoint remains individually measured and serialized. These rules
 * only decide whether two tensors describe the same numerical branch and can
 * therefore contribute to one elementwise layer-cosine aggregate.
 */

#pragma once

#include <string_view>

namespace llaminar2::test::parity
{
    /**
     * @brief Return whether a stage belongs in a layer cosine aggregate.
     *
     * Routing indices and weights have dedicated set/sparse-vector metrics and
     * never contribute cosine values. A raw routed-expert sum is also branch
     * dependent: when production and the reference select different expert
     * sets at the permitted low-weight top-k boundary, the two sums are not the
     * same elementwise operation. Its checkpoint remains visible in the stage
     * CSV, while the downstream combined MoE output and subsequent stages
     * continue to certify the semantic result.
     *
     * @param stage Semantic parity stage name.
     * @param routed_expert_set_exact Whether production and reference selected
     *        the same routed expert set for the current layer and row.
     * @return True only when the stage's cosine is mathematically comparable
     *         and should contribute to the layer aggregate.
     */
    constexpr bool parityStageContributesToLayerCosine(
        std::string_view stage,
        bool routed_expert_set_exact) noexcept
    {
        if (stage == "MOE_ROUTING_INDICES" ||
            stage == "MOE_ROUTING_WEIGHTS")
        {
            return false;
        }
        if (stage == "MOE_EXPERT_OUTPUT" &&
            !routed_expert_set_exact)
        {
            return false;
        }
        return true;
    }
} // namespace llaminar2::test::parity
