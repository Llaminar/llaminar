/**
 * @file ObservedCollectiveInventory.h
 * @brief Integration fixtures project the canonical observation, not fake capacity.
 *
 * Collective tests select real participants from DeviceManager's published
 * hardware. The shared production projector preserves UUIDs and reindexes
 * directed P2P edges for the selected set. No test-specific discovery protocol
 * or assumption that every GPU supports peer access is maintained here.
 */
#pragma once

#include "planning/ClusterInventoryGatherer.h"
#include <algorithm>
#include <limits>

namespace llaminar2::test
{
    /**
     * @brief Build one rank's real collective membership from canonical discovery.
     * @param cuda_count Maximum CUDA participants; zero excludes this backend.
     * @param rocm_count Maximum ROCm participants; zero excludes this backend.
     * @return Rank-local inventory carrying actual physical identity and links.
     */
    inline ClusterInventory observedLocalCollectiveInventory(
        int cuda_count = std::numeric_limits<int>::max(),
        int rocm_count = std::numeric_limits<int>::max())
    {
        if (cuda_count < 0 || rocm_count < 0)
            throw std::invalid_argument("Collective fixture participant limits must be non-negative");
        auto &manager = DeviceManager::instance();
        if (!manager.hardware()) manager.initialize(-1, false);
        std::vector<ComputeDevice> selected;
        for (const auto &device : manager.devices())
        {
            if (device.type == ComputeBackendType::GPU_CUDA && cuda_count > 0)
            {
                selected.push_back(device);
                --cuda_count;
            }
            if (device.type == ComputeBackendType::GPU_ROCM && rocm_count > 0)
            {
                selected.push_back(device);
                --rocm_count;
            }
        }
        ClusterInventory inventory;
        inventory.world_size = 1;
        inventory.ranks.push_back(makeRankInventory(*manager.hardware(), selected, {}));
        inventory.buildNodeAggregations();
        return inventory;
    }
}
