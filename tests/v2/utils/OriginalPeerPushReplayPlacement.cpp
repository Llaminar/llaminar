/**
 * @file OriginalPeerPushReplayPlacement.cpp
 * @brief Select real peer-capable participants through canonical MPI discovery.
 *
 * Visible ordinals need not share a NUMA owner or PCI bridge. Existing observed
 * peer grouping supplies the best complete cohort within an original rank;
 * other discovery ranks remain idle and retain their normal shutdown duties.
 */
#include "OriginalPeerPushReplayPlacement.h"
#include "utils/MPIContext.h"
#include "planning/ObservedPeerGrouping.h"
#include <iostream>

namespace llaminar2::test
{
    int originalPeerPushReplayWorldRank()
    {
        int rank = -1;
        if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS || rank < 0)
            throw std::runtime_error("Peer push replay lost its original MPI world rank");
        return rank;
    }

    std::vector<int> originalPeerPushReplayDevices(std::size_t degree, DeviceType backend,
                                                   std::string_view native_isa)
    {
        if (degree < 2 || (backend != DeviceType::CUDA && backend != DeviceType::ROCm))
            throw std::invalid_argument("Peer push replay requires at least two homogeneous GPU participants");
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        if (!inventory) throw std::invalid_argument("Peer push replay requires original MPI discovery");
        for (const auto &rank : inventory->ranks)
        {
            std::vector<DeviceId> eligible;
            for (const auto &gpu : rank.gpus)
                if (gpu.type == backend && rankOwnsGPUDevice(rank, gpu) &&
                    (native_isa.empty() || std::string_view(gpu.native_isa).substr(0, gpu.native_isa.find(':')) == native_isa))
                    eligible.emplace_back(gpu.type, gpu.local_device_id);
            if (eligible.size() < degree) continue;
            const ObservedPeerGrouping grouping(rank);
            const std::array sizes{degree};
            const auto group = grouping.select(eligible, sizes).front();
            if (grouping.coverage(group) != PeerAccessCoverage::Complete) continue;
            if (mpi->rank() != rank.rank) return {};
            std::vector<int> result;
            std::cout << "PEER_PUSH_OWNER,rank=" << rank.rank << ",numa=" << rank.cpu.numa_node << ",ordinals=";
            for (const auto device : group)
            {
                result.push_back(device.ordinal);
                std::cout << device.ordinal << ',';
            }
            std::cout << std::endl;
            return result;
        }
        throw std::invalid_argument("Peer push replay requires an original NUMA-owned complete P2P group for the selected backend");
    }
}
