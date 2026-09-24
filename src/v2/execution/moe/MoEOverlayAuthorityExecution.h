/**
 * @file MoEOverlayAuthorityExecution.h
 * @brief Typed execution location for the sole ExpertOverlay authority.
 *
 * Durable ownership always belongs to one ExpertOverlay epoch authority. This
 * type selects where that authority's histogram reduction, placement planning,
 * transfer publication, and epoch commit execute; it never selects a second
 * ownership protocol. The choice is frozen from routed-tier topology before
 * graph construction so captured graphs cannot silently change control planes.
 */

#pragma once

#include <cstdint>

namespace llaminar2
{
    /**
     * @brief Ownership locus for the sole live ExpertOverlay authority.
     *
     * This enum deliberately says nothing about tier count, vendor
     * homogeneity, collective choice, or controller fan-out. Those are
     * properties of the frozen controller topology. Authority ownership is
     * determined only by the owners of live routed-expert state: an all-GPU
     * overlay is device-resident, while any CPU participant makes the one
     * authority host-resident.
     */
    enum class MoEOverlayAuthorityExecutionKind : std::uint8_t
    {
        Unresolved = 0, ///< Declarative input not yet frozen against topology.
        HostResident, ///< At least one live routed-expert participant is CPU-owned.
        DeviceResident, ///< Every live routed-expert participant is GPU-owned.
    };

    /**
     * @brief Return the stable diagnostic spelling for an authority backend.
     * @param kind Typed execution backend.
     * @return Process-lifetime string literal.
     */
    [[nodiscard]] inline const char *toString(
        MoEOverlayAuthorityExecutionKind kind) noexcept
    {
        switch (kind)
        {
        case MoEOverlayAuthorityExecutionKind::Unresolved:
            return "unresolved";
        case MoEOverlayAuthorityExecutionKind::HostResident:
            return "host-resident";
        case MoEOverlayAuthorityExecutionKind::DeviceResident:
            return "device-resident";
        }
        return "unknown";
    }
} // namespace llaminar2
