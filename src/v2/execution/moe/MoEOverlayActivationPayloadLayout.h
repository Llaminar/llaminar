/**
 * @file MoEOverlayActivationPayloadLayout.h
 * @brief Typed hidden-matrix selection for node-local ExpertOverlay packets.
 *
 * A retained activation transaction has one immutable physical-row geometry.
 * That geometry selects both the hidden-matrix layout and its address family:
 * one-row transactions publish compact participant rows, while wider
 * transactions publish one physical-row matrix shared by every lane in the
 * rank-pair channel. Keeping the selection in this small device-safe type
 * prevents CPU and GPU endpoints from independently guessing how the same
 * authenticated packet bytes must be addressed.
 */

#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_PAYLOAD_HD __host__ __device__
#else
#define LLAMINAR_MOE_PAYLOAD_HD
#endif

namespace llaminar2
{
    /**
     * @brief Immutable interpretation of a dispatch packet's hidden matrix.
     *
     * The sparse metadata is compact in both cases. Only the hidden-row address
     * changes: compact row N addresses matrix row N, whereas a shared physical
     * payload addresses matrix row `row_ids[N]`.
     */
    enum class MoEOverlayActivationHiddenPayloadLayout : std::uint8_t
    {
        CompactRows = 0, ///< Matrix row N is compact packet row N.
        SharedPhysicalRows = 1, ///< Matrix row N is original physical row N.
    };

    /** @return Whether @p layout belongs to the node-local packet ABI. */
    [[nodiscard]] LLAMINAR_MOE_PAYLOAD_HD constexpr bool
    isValidMoEOverlayActivationHiddenPayloadLayout(
        MoEOverlayActivationHiddenPayloadLayout layout) noexcept
    {
        return layout ==
                   MoEOverlayActivationHiddenPayloadLayout::CompactRows ||
               layout == MoEOverlayActivationHiddenPayloadLayout::
                             SharedPhysicalRows;
    }

    /**
     * @brief Geometry-bound authority for one retained payload interpretation.
     *
     * Construction through @ref forPhysicalRows is the sole production
     * selection rule. The explicit layout remains in the value so captured
     * launch validation, CPU row addressing, diagnostics, and tests can reject
     * a mismatched or partially initialized selection.
     */
    struct MoEOverlayActivationPayloadSelection
    {
        std::int32_t physical_rows = 0; ///< Exact padded rows in the graph shape.
        MoEOverlayActivationHiddenPayloadLayout layout =
            MoEOverlayActivationHiddenPayloadLayout::CompactRows;

        /**
         * @brief Select the canonical payload interpretation for exact geometry.
         *
         * A non-positive value deliberately produces an invalid selection;
         * callers must reject it before graph construction or transaction
         * admission rather than repairing the geometry.
         */
        [[nodiscard]] LLAMINAR_MOE_PAYLOAD_HD static constexpr
            MoEOverlayActivationPayloadSelection
            forPhysicalRows(std::int32_t rows) noexcept
        {
            return {
                .physical_rows = rows,
                .layout = rows == 1
                              ? MoEOverlayActivationHiddenPayloadLayout::
                                    CompactRows
                              : MoEOverlayActivationHiddenPayloadLayout::
                                    SharedPhysicalRows,
            };
        }

        /** @return Whether geometry and layout form the canonical pair. */
        [[nodiscard]] LLAMINAR_MOE_PAYLOAD_HD constexpr bool valid()
            const noexcept
        {
            return physical_rows > 0 &&
                   isValidMoEOverlayActivationHiddenPayloadLayout(layout) &&
                   layout == forPhysicalRows(physical_rows).layout;
        }

        /** @return Whether the participant-local compact matrix is authoritative. */
        [[nodiscard]] LLAMINAR_MOE_PAYLOAD_HD constexpr bool usesCompactRows()
            const noexcept
        {
            return valid() &&
                   layout ==
                       MoEOverlayActivationHiddenPayloadLayout::CompactRows;
        }

        /** @return Whether the rank-pair physical-row matrix is authoritative. */
        [[nodiscard]] LLAMINAR_MOE_PAYLOAD_HD constexpr bool
        usesSharedPhysicalRows() const noexcept
        {
            return valid() &&
                   layout == MoEOverlayActivationHiddenPayloadLayout::
                                 SharedPhysicalRows;
        }

        bool operator==(
            const MoEOverlayActivationPayloadSelection &) const = default;
    };
} // namespace llaminar2

#undef LLAMINAR_MOE_PAYLOAD_HD
