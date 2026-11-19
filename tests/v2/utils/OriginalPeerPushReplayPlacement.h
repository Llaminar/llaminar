/**
 * @file OriginalPeerPushReplayPlacement.h
 * @brief Host-only original MPI/NUMA selection for the native peer-push replay.
 *
 * The shared HIP fixture consumes only original device ordinals. Canonical
 * discovery and peer grouping stay in a C++ translation unit, so their CPU
 * SIMD helpers never enter a GPU compilation pass.
 */
#pragma once
#include "backends/DeviceType.h"
#include <cstddef>
#include <vector>
#include <string_view>

namespace llaminar2::test
{
    /** @return Exact physical owner rank in the original MPI discovery session. */
    int originalPeerPushReplayWorldRank();

    /**
     * @brief Select a complete peer group owned by one original discovery rank.
     * @param degree Requested native participant count, at least two.
     * @param backend Requested homogeneous GPU backend; CPU is rejected.
     * @param native_isa Optional native target of an architecture-specific test.
     * @return Original device ordinals, or empty on an unselected original rank.
     * @throws std::invalid_argument For malformed or unavailable native ownership.
     * Every original rank calls once; no MPI scope or CPU binding is replaced.
     */
    std::vector<int> originalPeerPushReplayDevices(std::size_t degree,
                                                  DeviceType backend = DeviceType::ROCm,
                                                  std::string_view native_isa = {});
}
