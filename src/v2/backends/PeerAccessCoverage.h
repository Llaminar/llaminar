/**
 * @file PeerAccessCoverage.h
 * @brief Lightweight typed classification of a GPU peer-access domain.
 *
 * This policy type is kept separate from the legacy compute-context header so
 * declarative graph configuration can name its transport decision without
 * pulling device-manager and backend-context interfaces into every model
 * translation unit.
 */

#pragma once

namespace llaminar2
{
    /** Driver-reported peer-access coverage within one homogeneous GPU set. */
    enum class PeerAccessCoverage
    {
        None,    ///< No requested device pair exposes direct peer access.
        Partial, ///< At least one, but not every, pair exposes both directions.
        Complete ///< Every requested device pair exposes both directions.
    };

    /**
     * @brief Return a stable diagnostic name for peer-access coverage.
     * @param coverage Coverage value to format.
     * @return Lower-case name suitable for logs and PerfStats tags.
     */
    [[nodiscard]] constexpr const char *peerAccessCoverageName(
        PeerAccessCoverage coverage) noexcept
    {
        switch (coverage)
        {
        case PeerAccessCoverage::None:
            return "none";
        case PeerAccessCoverage::Partial:
            return "partial";
        case PeerAccessCoverage::Complete:
            return "complete";
        }
        return "invalid";
    }
} // namespace llaminar2
