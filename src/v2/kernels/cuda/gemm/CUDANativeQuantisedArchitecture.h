/**
 * @file CUDANativeQuantisedArchitecture.h
 * @brief Project CUDA verifier dispatch from the original physical inventory.
 * No native query, context change or secondary inventory occurs here.
 */
#pragma once

#include "backends/cuda/CUDAHardwareMetadata.h"
#include "kernels/common/VerifierDispatchTable.h"

namespace llaminar2::cuda
{
    /** @brief Project already admitted physical facts into the native quantized policy.
     * @param facts The immutable CUDA inventory record for this participant.
     * @return Complete observed architecture, ordinal, warp width and resource limits.
     * @throws std::invalid_argument If the observation is missing or contradictory.
     */
    inline native_quantised::NativeQuantisedGPUHardware nativeQuantisedHardware(const CUDADeviceHardware &facts)
    {
        if (facts.sm_major <= 0 || facts.sm_minor < 0)
            throw std::invalid_argument("CUDA verifier dispatch requires an observed SM version");
        native_quantised::NativeQuantisedGPUHardware result{
            native_quantised::NativeQuantisedArchitecture::cuda(facts.sm_major, facts.sm_minor),
            facts.ordinal, facts.sm_count, facts.native_warp_width,
            facts.maximum_threads_per_block, facts.shared_memory_per_block};
        result.requireObserved();
        return result;
    }

    /** @brief Borrow architecture facts from the exact CUDA participant.
     * @param ordinal Native ordinal admitted by the CUDA hardware owner.
     * @return Original immutable participant facts.
     */
    inline native_quantised::NativeQuantisedGPUHardware nativeQuantisedHardwareForDevice(int ordinal)
    {
        return nativeQuantisedHardware(cudaHardwareMetadata().at(ordinal));
    }

    /** @brief Two-row DP4A ownership is the CUDA portable grouped verifier schedule. */
    inline constexpr int portableGroupedVerifierRows = 2;
}
