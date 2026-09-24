/**
 * @file PlanningObservedResource.h
 * @brief Project one discovered allocator into the canonical physical BOM.
 *
 * This projection neither queries live capacity nor admits or subtracts bytes.
 * Startup measurement contributors share it so a rank-local ordinal cannot
 * accidentally acquire another rank's resource or an invented capacity reserve.
 */
#pragma once
#include "PhysicalMemoryAuthority.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include <algorithm>
#include <stdexcept>

namespace llaminar2
{
    /** @return Exact published allocator; unknown and noncanonical host devices are rejected. */
    inline PhysicalMemoryResource planningObservedResource(const RankInventory &rank, DeviceId device)
    {
        const DeviceInfo *info = &rank.cpu;
        if (device.is_gpu())
        {
            const auto found = std::find_if(rank.gpus.begin(), rank.gpus.end(), [&](const auto &gpu) {
                return gpu.type == device.type && gpu.local_device_id == device.ordinal;
            });
            if (found == rank.gpus.end()) throw std::invalid_argument("Planning allocator absent from rank inventory");
            info = &*found;
        }
        else if (device != DeviceId::cpu())
            throw std::invalid_argument("Planning host allocator must use the rank-bound CPU resource");
        return {rank.rank, device, info->memory_bytes, info->free_memory_bytes};
    }
}
