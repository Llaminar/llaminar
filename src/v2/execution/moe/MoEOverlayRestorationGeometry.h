/**
 * @file MoEOverlayRestorationGeometry.h
 * @brief Checked original logical geometry for cold prepared-context restoration.
 *
 * Rank-wide and CPU host teardown share one finite wave bound. The snapshot's
 * typed complete ownership arm supplies its logical layers/experts; whole-only
 * projections and physical fragment populations cannot select this geometry.
 * This helper owns no live progress, memory, policy or placement state.
 */
#pragma once
#include "MoEOverlayResidencyAuthority.h"
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Bound nonempty restoration waves by the original logical assignments.
     * @param snapshot Valid original current publication, including complete CPU cohorts.
     * @return Logical layers times experts plus the existing terminal allowance.
     * @throws std::logic_error For absent/invalid complete original ownership geometry.
     * @throws std::overflow_error If its finite bound cannot be represented.
     *
     * Each nonempty closed-cycle wave fixes at least one original assignment.
     * Existing wave/shadow budgets remain unchanged. Fragment members are not
     * additional logical experts, and no counter is reconstructed or retained.
     */
    [[nodiscard]] inline std::size_t preparedPlacementRestorationWaveLimit(
        const MoEOverlayResidencySnapshot &snapshot)
    {
        if (!snapshot.valid())
            throw std::logic_error("Prepared restoration progress bound requires a valid original placement snapshot");
        const auto layers = snapshot.layerCount();
        const auto experts = snapshot.expertCount();
        if (layers <= 0 || experts <= 0)
            throw std::logic_error("Prepared restoration progress bound requires complete original logical geometry");
        if (std::size_t(layers) > (std::numeric_limits<std::size_t>::max() - 1u) / std::size_t(experts))
            throw std::overflow_error("Prepared restoration progress bound exceeds original logical geometry");
        return std::size_t(layers) * std::size_t(experts) + 1u;
    }
}
