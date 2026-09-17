/**
 * @file RetainedParentPerfEvidence.h
 * @brief Common diagnostic projection of a retained parent's physical boundary.
 *
 * First-use submission is emitted by the capture controller, while setup-only
 * materialization and later submissions are emitted by the executor. They must
 * describe the same child and concurrent-service inventory. This pure helper
 * retains no state and cannot authorize execution; callers supply the actual
 * composition inputs or the sealed replay plan, never logical MoE layer counts.
 */
#pragma once

#include "../../../utils/PerfStatsCollector.h"

#include <cstddef>
#include <span>
#include <string>

namespace llaminar2
{
    /**
     * @brief Serialize one physical parent inventory identically at every lifecycle edge.
     * @param context Graph family's stable diagnostic identity.
     * @param child_units Actual graph-only units owned by the composition.
     * @param service_segments Actual concurrently serviced manual-segment indices.
     * @return Complete boundary tags; callers may add event-specific diagnostics.
     *
     * A service program can contain many logical CPU cutpoints. Counting its
     * physical segment span keeps first launch and replay consistent without
     * inspecting mutable device state or introducing any synchronization.
     */
    [[nodiscard]] inline PerfStatsCollector::Tags retainedParentBoundaryTags(
        const std::string &context,
        std::size_t child_units,
        std::span<const std::size_t> service_segments)
    {
        return {{"context", context},
                {"child_units", std::to_string(child_units)},
                {"boundary_authority", service_segments.empty()
                    ? "captured_device_units" : "concurrent_ticket_service"},
                {"ticket_service_units", std::to_string(service_segments.size())}};
    }
}
