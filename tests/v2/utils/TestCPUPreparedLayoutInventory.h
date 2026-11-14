/**
 * @file TestCPUPreparedLayoutInventory.h
 * @brief Device-free uniform-canonical CPU inventory for live-observation admission fixtures.
 *
 * Production overlay admission derives the uniform CPU prepared layout from the
 * live inventory (same observation and cross-layout refusal the hardware binder
 * applies). Device-free admission fixtures therefore supply a minimal
 * inventory whose ranks publish the build's canonical layout — the fixture
 * builds its plan like production instead of hand-setting a field.
 */
#pragma once

#include "execution/mpi_orchestration/DeviceInventory.h"
#include <initializer_list>

namespace llaminar2::test
{
    /** @return Minimal uniform-canonical inventory naming @p ranks. */
    inline ClusterInventory cpuUniformCanonicalInventory(
        std::initializer_list<int> ranks)
    {
        ClusterInventory inventory;
        inventory.world_size = static_cast<int>(ranks.size());
        for (const int rank : ranks)
        {
            RankInventory observed;
            observed.rank = rank;
            observed.cpu_prepared_layout =
                cpu::native_quantised::kHostPreparedLayout;
            inventory.ranks.push_back(std::move(observed));
        }
        return inventory;
    }
} // namespace llaminar2::test
