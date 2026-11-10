/**
 * @file ROCmNativeQuantisedArchitecture.h
 * @brief Borrow actual ROCm dispatch facts from the admitted immutable inventory.
 * No native property query, device switch, allocation or shadow inventory occurs.
 */
#pragma once
#include "kernels/common/NativeQuantisedArchitecture.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
namespace llaminar2::rocm
{
    /** @brief Project the exact prepared participant before architecture-aware lookup.
     * @param ordinal Original native ordinal, never a visibility-array position guess.
     * @return Actual cached ISA, wave and launch-resource limits.
     */
    inline native_quantised::NativeQuantisedGPUHardware nativeQuantisedHardwareForDevice(int ordinal)
    {
        const auto &facts = rocmHardwareMetadata().at(ordinal);
        return {native_quantised::NativeQuantisedArchitecture::fromROCmISATarget(facts.isa_target),
            facts.ordinal, facts.compute_units, facts.native_wave_width, facts.maximum_threads_per_block, facts.lds_per_block};
    }
}
