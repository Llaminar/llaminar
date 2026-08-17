/**
 * @file MoEOverlayRetainedParentComposer.h
 * @brief Shared retained-parent discovery for ExpertOverlay endpoint graphs.
 *
 * Model graphs declare packet authority through typed activation stages. This
 * helper translates that declarative role into the one callback understood by
 * DeviceGraphExecutor. Decode, prefill, continuation, and follower runners all
 * use this same entry point, preventing any execution phase from silently
 * reverting to a host-walked sparse schedule.
 */

#pragma once

#include "../local_execution/graph/DeviceGraphExecutor.h"

#include <optional>

namespace llaminar2
{
    class ComputeGraph;

    /**
     * @brief Derive one topology-composed parent callback from typed packet stages.
     *
     * A continuation graph owns dispatch-pack and return-consume frontiers; a
     * follower graph owns dispatch-consume and return-pack frontiers. A graph
     * without either role returns `std::nullopt`. Mixed or half-declared roles
     * are rejected because they cannot represent one physical endpoint.
     *
     * The returned callback borrows no graph or cache state. At capture setup it
     * receives the executor-owned child views and builds one native parent on
     * the destination's exact stream.
     *
     * @param graph Declarative per-device graph whose packet role is inspected.
     * @return Empty for an ordinary graph, otherwise its reusable parent composer.
     * @throws std::runtime_error for mixed or incomplete packet authority.
     */
    [[nodiscard]] std::optional<
        DeviceGraphExecutor::RetainedParentCompositionHook>
    makeMoEOverlayRetainedParentComposer(const ComputeGraph &graph);
} // namespace llaminar2
