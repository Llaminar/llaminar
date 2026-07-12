/**
 * @file VerifierRowTestInventory.h
 * @brief Canonical runtime-M inventories for grouped verifier regression tests.
 *
 * Grouped MTP kernels must not encode speculative depth as a four-row ABI.
 * The default correctness inventory therefore covers every multi-row verifier
 * count from two through sixteen, corresponding to fifteen draft rows plus the
 * target bonus row. A final M=31 case exceeds that default graph capacity and
 * catches accidental admission limits in otherwise runtime-sized kernels.
 *
 * Expensive real-model-shape tests may use the boundary inventory, but every
 * backend and tensor format still needs at least one canonical suite that uses
 * the complete runtime inventory.
 */

#pragma once

#include <array>

namespace llaminar2::test
{
    /**
     * @brief Complete default certification range plus an extended-depth probe.
     */
    inline constexpr std::array<int, 16> kGroupedVerifierRuntimeRows = {
        2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 31};

    /**
     * @brief Tile and capacity boundaries for expensive production-size cases.
     */
    inline constexpr std::array<int, 6> kGroupedVerifierBoundaryRows = {
        2, 3, 4, 5, 16, 31};
} // namespace llaminar2::test
