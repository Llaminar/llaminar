/**
 * @file VerifierRowTestInventory.h
 * @brief Canonical runtime-M inventories for grouped verifier regression tests.
 *
 * Grouped MTP kernels must not encode speculative depth as a four-row ABI.
 * The default correctness inventory therefore covers every multi-row verifier
 * count from two through sixteen, corresponding to fifteen draft rows plus the
 * target bonus row. M=17 is the first grouped-prefill row outside that MTP
 * contract and therefore guards backend engine-selection boundaries. A final
 * M=31 case exceeds the default graph capacity and catches accidental admission
 * limits in otherwise runtime-sized kernels.
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
    inline constexpr std::array<int, 17> kGroupedVerifierRuntimeRows = {
        2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 31};

    /**
     * @brief Runtime-M inventory plus the first scalable Qwen MoE route group.
     *
     * Qwen3.6 routes eight experts per token. M=31 therefore publishes 248
     * route slots and remains inside the one-block runtime planner's 256-slot
     * capacity. M=33 is the smallest odd row count that crosses that boundary,
     * so production-shape all-format tests use this inventory to prove the
     * scalable count/scan/scatter path as well as the verifier-sized path.
     */
    inline constexpr std::array<int, 18> kGroupedVerifierQwenScalableRows = {
        2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 31, 33};

    /**
     * @brief Tile and capacity boundaries for expensive production-size cases.
     */
    inline constexpr std::array<int, 7> kGroupedVerifierBoundaryRows = {
        2, 3, 4, 5, 16, 17, 31};
} // namespace llaminar2::test
