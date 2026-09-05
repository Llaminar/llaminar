/**
 * @file LocalTPCollectiveInventory.cpp
 * @brief Pure projection of LocalTP participants into collective inventory.
 */

#include "LocalTPCollectiveInventory.h"

#include <set>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    ClusterInventory buildLocalTPCollectiveInventory(
        const std::vector<GlobalDeviceAddress> &participants,
        int mpi_rank,
        int mpi_world_size)
    {
        if (participants.empty())
        {
            throw std::invalid_argument(
                "buildLocalTPCollectiveInventory: participants cannot be empty");
        }
        if (mpi_world_size <= 0 || mpi_rank < 0 || mpi_rank >= mpi_world_size)
        {
            throw std::invalid_argument(
                "buildLocalTPCollectiveInventory: invalid MPI rank/world size");
        }

        RankInventory rank_inventory;
        rank_inventory.rank = mpi_rank;
        rank_inventory.node_id = 0;
        rank_inventory.local_rank = mpi_rank;
        rank_inventory.hostname = "localhost";
        rank_inventory.cpu.type = DeviceType::CPU;
        rank_inventory.cpu.local_device_id = 0;

        std::set<std::pair<DeviceType, int>> identities;
        for (const GlobalDeviceAddress &participant : participants)
        {
            if (participant.device_ordinal < 0)
            {
                throw std::invalid_argument(
                    "buildLocalTPCollectiveInventory: negative device ordinal for " +
                    participant.toString());
            }

            const auto identity =
                std::make_pair(participant.device_type, participant.device_ordinal);
            if (!identities.insert(identity).second)
            {
                throw std::invalid_argument(
                    "buildLocalTPCollectiveInventory: duplicate local device " +
                    participant.toString());
            }

            if (participant.isCPU())
            {
                // RankInventory has one CPU record. NUMA ownership remains in
                // the LocalTP participant address and is not needed for routing.
                rank_inventory.cpu.numa_node = participant.numa_node;
                continue;
            }
            if (!participant.isCUDA() && !participant.isROCm())
            {
                throw std::invalid_argument(
                    "buildLocalTPCollectiveInventory: unsupported collective device " +
                    participant.toString());
            }

            DeviceInfo gpu;
            gpu.type = participant.device_type;
            gpu.local_device_id = participant.device_ordinal;
            gpu.numa_node = participant.numa_node;
            gpu.name = participant.toShortString();
            rank_inventory.gpus.push_back(std::move(gpu));
        }

        ClusterInventory inventory;
        inventory.world_size = mpi_world_size;
        inventory.ranks.push_back(std::move(rank_inventory));
        inventory.buildNodeAggregations();
        return inventory;
    }

} // namespace llaminar2
