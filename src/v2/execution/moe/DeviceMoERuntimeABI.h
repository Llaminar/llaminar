/**
 * @file DeviceMoERuntimeABI.h
 * @brief Lightweight ABI contract for graph-resident MoE layer state.
 *
 * GPU translation units intentionally use a device-friendly view of
 * DeviceMoELayerRuntime instead of including MoERuntimeTable.h, whose host
 * tensor dependencies contain x86 SIMD code. These constants make that split
 * mechanical rather than conventional: the host definition and every CUDA/HIP
 * view must independently prove the same 64-bit size and critical offsets.
 */
#pragma once

#include <cstddef>

namespace llaminar2::moe_runtime_abi
{
    inline constexpr std::size_t kLayerRuntimeBytes = 89536;
    inline constexpr std::size_t kRouteParticipantIdsOffset = 89360;
    inline constexpr std::size_t kDeferredVerifierExpertIdsOffset = 89368;
    inline constexpr std::size_t kDeferredVerifierParticipantIdsOffset = 89376;
    inline constexpr std::size_t kExpertCountsOffset = 89384;
    inline constexpr std::size_t kDeferredVerifierRouteCapacityOffset = 89512;
    inline constexpr std::size_t kParticipantCountOffset = 89520;
    inline constexpr std::size_t kCurrentBatchLLEPMovementObservedOffset = 89524;
    inline constexpr std::size_t
        kCurrentBatchLLEPNonOwnerAssignmentObservedOffset = 89528;

    static_assert(sizeof(void *) == 8,
                  "DeviceMoELayerRuntime ABI requires 64-bit pointers");
} // namespace llaminar2::moe_runtime_abi
