/**
 * @file MoEOverlayReturnLayout.h
 * @brief Shared arithmetic identity for host MoE return planning and transport.
 *
 * Capacity planning and publication use the same typed layout: token partials
 * need one row per token; canonical publication needs one row per live expert
 * route. A consumer must reject another layout rather than reinterpret bytes.
 */
#pragma once
#include <cstdint>

namespace llaminar2
{
    /** @brief Arithmetic and row-identity contract carried by a return packet. */
    enum class MoEOverlayReturnLayout : uint32_t
    {
        ParticipantTokenPartials = 0, ///< Weighted participant sums indexed by token row.
        CanonicalExpertRoutes = 1, ///< Raw expert rows indexed by original flat router slot.
    };

    /** @return Whether a wire value names an installed return arithmetic contract. */
    [[nodiscard]] constexpr bool isValidMoEOverlayReturnLayout(MoEOverlayReturnLayout layout) noexcept
    {
        return layout == MoEOverlayReturnLayout::ParticipantTokenPartials ||
               layout == MoEOverlayReturnLayout::CanonicalExpertRoutes;
    }
}
