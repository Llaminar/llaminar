/**
 * @file PlanningPeerCopyAccess.h
 * @brief Original discovery facts selecting directed physical-copy measurements.
 *
 * A destination-owned peer copy reads source memory, so its access bit is the
 * reverse coordinate of the requested data movement. Missing observations
 * cannot authorize direct access. This pure boundary never probes a driver or
 * changes the production collective transport.
 */
#pragma once
#include "planning/ObservedPeerGrouping.h"
#include <cstdint>
#include <span>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Project an original discovery matrix onto a homogeneous local group.
     * @param rank Immutable original process/NUMA ownership and complete peer facts.
     * @param devices Exact physical device order retained by the sample receipt.
     * @return Row-major source-to-destination copy eligibility, with zero self edges.
     * @throws std::invalid_argument For absent, malformed or foreign discovery facts.
     */
    inline std::vector<std::uint8_t> planningDirectedPeerCopyAccess(
        const RankInventory &rank, std::span<const DeviceId> devices)
    {
        const ObservedPeerGrouping facts(rank);
        (void)facts.coverage(devices); // Validates full discovery geometry and original ownership.
        const auto type = devices.front().type;
        const auto &observed = type == DeviceType::CUDA ? rank.p2p_cuda : rank.p2p_rocm;
        const auto count = type == DeviceType::CUDA ? rank.p2p_cuda_count : rank.p2p_rocm_count;
        std::vector<DeviceId> discovered;
        for (const auto &device : rank.gpus)
            if (device.type == type) discovered.emplace_back(type, device.local_device_id);
        std::vector<std::uint8_t> result(devices.size() * devices.size(), 0);
        for (size_t source = 0; source < devices.size(); ++source)
            for (size_t destination = 0; destination < devices.size(); ++destination)
                if (source != destination)
                {
                    const auto from = std::find(discovered.begin(), discovered.end(), devices[source]) - discovered.begin();
                    const auto to = std::find(discovered.begin(), discovered.end(), devices[destination]) - discovered.begin();
                    result[source * devices.size() + destination] = observed[size_t(to) * count + size_t(from)] != 0;
                }
        return result;
    }
}
