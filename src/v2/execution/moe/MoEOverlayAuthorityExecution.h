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
    /** @brief Physical execution backend for ExpertOverlay authority work. */
    enum class MoEOverlayAuthorityExecutionKind : std::uint8_t
    {
        Unresolved = 0, ///< Declarative input not yet frozen against topology.
        HostCoordinated, ///< Multiple tiers or heterogeneous tier participants.
        HomogeneousDeviceResident, ///< One tier whose participants share one device type.
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
        case MoEOverlayAuthorityExecutionKind::HostCoordinated:
            return "host-coordinated";
        case MoEOverlayAuthorityExecutionKind::HomogeneousDeviceResident:
            return "homogeneous-device-resident";
        }
        return "unknown";
    }
} // namespace llaminar2
