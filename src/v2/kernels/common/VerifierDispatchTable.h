/**
 * @file VerifierDispatchTable.h
 * @brief Select grouped verifier launch schedules from the owning GPU's admitted hardware.
 *
 * Portable schedules belong to each backend. Measured tables override scheduling
 * for qualified devices while retaining that backend's arithmetic and row contract.
 * Selection happens when recording or preparing a launch. The device live count
 * never participates in table identity.
 */
#pragma once

#include "kernels/common/NativeQuantisedArchitecture.h"

namespace llaminar2::native_quantised {

/** @brief Source of verifier launch parameters for a valid native GPU. */
enum class VerifierDispatchTable { Portable, MI50, SM86 };

/**
 * @brief Name the schedule that recorded a grouped projection in PerfStats.
 * @param table Table selected from the participant's observed hardware.
 * @return Stable provenance tag shared by both GPU backends.
 * @throws std::invalid_argument For an invalid table value.
 */
[[nodiscard]] constexpr const char* verifierDispatchTableName(VerifierDispatchTable table)
{
    switch (table) {
    case VerifierDispatchTable::Portable: return "portable";
    case VerifierDispatchTable::MI50: return "gfx906-mi50";
    case VerifierDispatchTable::SM86: return "sm86";
    }
    throw std::invalid_argument("Unknown grouped verifier dispatch table");
}

/**
 * @brief Resolve a measured table when the actual participant meets its hardware contract.
 * @param hardware Immutable observation from the owning CUDA or ROCm inventory.
 * @return The measured table, or the backend's portable table for an unmeasured device.
 * @throws std::invalid_argument If the GPU observation is absent or contradictory.
 */
[[nodiscard]] constexpr VerifierDispatchTable selectVerifierDispatchTable(const NativeQuantisedGPUHardware& hardware)
{
    hardware.requireObserved();
    if (hardware.architecture == NativeQuantisedArchitecture::rocm(9, 0, 6) &&
        hardware.compute_units == 60 && hardware.native_lane_width == 64)
        return VerifierDispatchTable::MI50;
    if (hardware.architecture == NativeQuantisedArchitecture::cuda(8, 6))
        return VerifierDispatchTable::SM86;
    return VerifierDispatchTable::Portable;
}

} // namespace llaminar2::native_quantised
