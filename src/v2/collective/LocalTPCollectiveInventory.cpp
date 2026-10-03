/**
 * @file LocalTPCollectiveInventory.cpp
 * @brief Pure projection of LocalTP participants into collective inventory.
 */

#include "LocalTPCollectiveInventory.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    PeerAccessCoverage localTPPeerAccessCoverage(const RankInventory &inventory, const std::vector<DeviceId> &devices)
    {
        if (devices.size() < 2 || !devices.front().is_gpu())
            throw std::invalid_argument("LocalTP P2P coverage needs a homogeneous GPU subset");
        const auto type = devices.front().type;
        const auto &matrix = type == DeviceType::CUDA ? inventory.p2p_cuda : inventory.p2p_rocm;
        const int count = type == DeviceType::CUDA ? inventory.p2p_cuda_count : inventory.p2p_rocm_count;
        std::vector<int> ordinals, indices;
        for (const auto &gpu : inventory.gpus) if (gpu.type == type)
        {
            if (gpu.local_device_id < 0 || std::find(ordinals.begin(), ordinals.end(), gpu.local_device_id) != ordinals.end())
                throw std::invalid_argument("LocalTP P2P inventory has repeated or invalid GPU identities");
            ordinals.push_back(gpu.local_device_id);
        }
        if (count < 2 || static_cast<std::size_t>(count) != ordinals.size() ||
            matrix.size() / static_cast<std::size_t>(count) != static_cast<std::size_t>(count) ||
            matrix.size() % static_cast<std::size_t>(count))
            throw std::invalid_argument("LocalTP P2P coverage lacks a complete observed matrix");
        for (auto device : devices)
        {
            const auto found = std::find(ordinals.begin(), ordinals.end(), device.ordinal);
            if (device.type != type || found == ordinals.end())
                throw std::invalid_argument("LocalTP P2P subset names a foreign endpoint");
            const auto index = static_cast<int>(found - ordinals.begin());
            if (std::find(indices.begin(), indices.end(), index) != indices.end())
                throw std::invalid_argument("LocalTP P2P subset repeats an endpoint");
            indices.push_back(index);
        }
        bool any = false, all = true;
        for (int from : indices) for (int to : indices) if (from != to)
        { const bool enabled = matrix[static_cast<std::size_t>(from) * count + to]; any |= enabled; all &= enabled; }
        return all ? PeerAccessCoverage::Complete : any ? PeerAccessCoverage::Partial : PeerAccessCoverage::None;
    }

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
