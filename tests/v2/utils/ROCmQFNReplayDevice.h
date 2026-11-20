/**
 * @file ROCmQFNReplayDevice.h
 * @brief Select the physical MI50 participant required by the QFN ROCm replay gates.
 *
 * Mixed GPU hosts can expose a gfx1100 device before the MI50 cohort. Select
 * the measured ISA and CU geometry from native metadata before binding the
 * test stream, device buffers and graph capture to that same participant.
 */
#pragma once

#include "backends/rocm/ROCmHardwareMetadata.h"

namespace llaminar2::test
{
    /**
     * @brief Find the first visible 60-CU gfx906 wave64 participant.
     * @return Native device ordinal or -1 when that geometry is absent.
     */
    inline int findQFNMI50ReplayDevice()
    {
        initializeROCmHardwareMetadata();
        const auto &inventory = rocmHardwareMetadata();
        for (int ordinal = 0; ordinal < static_cast<int>(inventory.size()); ++ordinal)
        {
            const auto &device = inventory.at(ordinal);
            if ((device.isa_target == "gfx906" || device.isa_target.starts_with("gfx906:")) &&
                device.compute_units == 60 && device.native_wave_width == 64)
                return ordinal;
        }
        return -1;
    }
}
