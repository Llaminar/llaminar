/**
 * @file MoEOverlayRetainedParentComposer.h
 * @brief Shared retained-parent discovery for ExpertOverlay endpoint graphs.
 *
 * Model graphs declare packet authority through typed activation stages and a
 * graph-wide heterogeneous ticket envelope. This helper translates those
 * declarative roles into the one callback understood by DeviceGraphExecutor.
 * Decode, prefill, continuation, and follower runners all use this same entry
 * point, preventing any execution phase from silently reverting to a
 * host-walked sparse schedule.
 */

#pragma once

#include "../local_execution/graph/DeviceGraphExecutor.h"

#include <optional>

namespace llaminar2
{
    class ComputeGraph;

    /**
     * @brief Complete replay authority selected from one ExpertOverlay graph.
     *
     * Keeping policy and composer together prevents callers from pairing a
     * topology-aware lowerer with the wrong host lifecycle. The ordinary packet
     * form contains graph-only children. A heterogeneous authority contains the
     * same child parent plus typed concurrent CPU service; its same-domain
     * followers retain only their ordered graph children.
     */
    struct MoEOverlayRetainedParentPlan
    {
        DeviceGraphExecutor::GraphReplayPlanPolicy replay_policy =
            DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph;
        DeviceGraphExecutor::RetainedParentCompositionHook composer;

        /** @return Whether this object names one supported retained lifecycle. */
        [[nodiscard]] bool valid() const noexcept
        {
            return DeviceGraphExecutor::isRetainedParentPlanPolicy(
                       replay_policy) &&
                   static_cast<bool>(composer);
        }
    };

    /**
     * @brief Derive one topology-composed parent from typed endpoint ownership.
     *
     * Heterogeneous ticket graphs always receive an ordered stage-owned parent:
     * scalar and batched packet stages retain their complete timeline DAG in
     * their captured children, while a same-domain follower may have no packet
     * stage at all. The parent spans only the real host-ticket cutpoints. For an
     * ordinary scalar graph, the explicit continuation/follower lowerer owns
     * the packet frontiers. A graph without one of these roles returns
     * `std::nullopt`; mixed or half-declared roles are rejected.
     *
     * The returned callback borrows no graph or cache state. At capture setup it
     * receives the executor-owned child views and builds one native parent on
     * the destination's exact stream.
     *
     * @param graph Declarative per-device graph whose packet role is inspected.
     * @return Empty for an ordinary graph, otherwise its reusable parent composer.
     * @throws std::runtime_error for mixed or incomplete packet authority.
     */
    [[nodiscard]] std::optional<MoEOverlayRetainedParentPlan>
    makeMoEOverlayRetainedParentPlan(const ComputeGraph &graph);
} // namespace llaminar2
