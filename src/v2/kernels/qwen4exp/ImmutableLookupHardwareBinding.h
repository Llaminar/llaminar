/**
 * @file ImmutableLookupHardwareBinding.h
 * @brief Setup-only physical clock binding for the existing immutable-source deadline.
 *
 * ROCm's wall_clock64 ticks use the admitted constant-frequency wall clock.
 * Source/coordinator generations and mutable lookup state remain independently
 * owned; this helper binds only the unchanged standard thirty-second deadline.
 */
#pragma once
#include "backends/rocm/ROCmHardwareMetadata.h"
#include <cstdint>

namespace llaminar2::qwen4exp
{
    /**
     * @brief Convert the exact admitted ROCm wall clock into the standard source deadline.
     * @param hardware Exact source-owning visible device's immutable record.
     * @return Thirty seconds of wall_clock64 ticks (kilohertz times 30,000 milliseconds).
     * @throws std::invalid_argument For an absent or invalid physical clock binding.
     */
    [[nodiscard]] inline std::uint64_t rocmImmutableLookupTimeoutTicks(const ROCmDeviceHardware &hardware)
    {
        return rocmServiceWallClockRateKHz(hardware) * 30'000u;
    }
}
