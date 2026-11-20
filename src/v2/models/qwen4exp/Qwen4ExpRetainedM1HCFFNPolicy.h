/**
 * @file Qwen4ExpRetainedM1HCFFNPolicy.h
 * @brief Exact cold workload routing policy for the experimental retained M1 region.
 */
#pragma once

#include "execution/local_execution/graph/IGraphBuilder.h"

namespace llaminar2 {
/**
 * @brief Select the original main inference, single-request decode family.
 * @param opt_in Explicit experimental selection at model ownership.
 * @param phase Original graph construction phase.
 * @param role Original complete graph execution role.
 * @param rows Admitted sequence width.
 * @param requests Admitted request count in the graph.
 * @return Whether the experimental reader participates in this graph.
 */
constexpr bool useQwen4ExpRetainedM1HCFFN(bool opt_in, ForwardExecutionPhase phase,
    ForwardExecutionRole role, int rows, int requests) noexcept
{
    return opt_in&&phase==ForwardExecutionPhase::Decode&&role==ForwardExecutionRole::MainInference&&
        rows==1&&requests==1;
}
}
