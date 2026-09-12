/**
 * @file MTPConditionForwardPurpose.h
 * @brief Commit ownership of a captured scalar main-model MTP condition.
 *
 * A pending condition from a speculative result has already been charged by
 * accepted-state publication. A budget-one or forced token instead commits a
 * new serial row. Both run identical model arithmetic, but only the latter
 * owns a new maintenance-clock edge. Keep that distinction explicit from the
 * public runner through every participant and the retained graph identity.
 */
#pragma once

#include <array>
#include <cstdint>

namespace llaminar2
{
    /** @brief Exactly one owner of a condition row's decode commit. */
    enum class MTPConditionForwardPurpose : std::uint8_t
    {
        SpeculativeContinuation, ///< Accepted-result publication owns cadence.
        CommittedSerialToken, ///< This complete forward owns one new commit.
    };

    /** Complete setup inventory, also consumed by native-memory admission. */
    inline constexpr std::array kMTPConditionForwardPurposes{
        MTPConditionForwardPurpose::SpeculativeContinuation,
        MTPConditionForwardPurpose::CommittedSerialToken};
}
