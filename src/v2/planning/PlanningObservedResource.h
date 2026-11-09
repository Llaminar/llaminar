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
    /**
     * @brief Attach the discovered physical backing of one GPU allocator.
     *
     * Every planner that projects a GPU from a rank inventory routes through
     * this function, so the same allocator never appears with two backings
     * (PhysicalMemoryPlanBuilder rejects that as conflicting observations).
     * For an integrated GPU the uncapped driver free figure is a sample of the
     * rank host pool and becomes the joint pool limit; the resource's own
     * admission availability keeps any user cap.
     * @throws std::invalid_argument when a GPU is absent from the inventory.
     */
    inline PhysicalMemoryResource withObservedBacking(PhysicalMemoryResource resource, const RankInventory &rank)
    {
        if (!resource.device.is_gpu())
            return resource;
        const auto found = std::find_if(rank.gpus.begin(), rank.gpus.end(), [&](const auto &gpu) {
            return gpu.type == resource.device.type && gpu.local_device_id == resource.device.ordinal;
        });
        if (found == rank.gpus.end())
            throw std::invalid_argument("Physical backing requested for a GPU absent from rank inventory");
        resource.backing = found->memory_backing;
        resource.pool_available_bytes =
            found->memory_backing == PhysicalMemoryBacking::RankHostPool ? found->free_memory_bytes : 0u;
        return resource;
    }

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
        return withObservedBacking({rank.rank, device, info->memory_bytes, info->free_memory_bytes}, rank);
    }
}
