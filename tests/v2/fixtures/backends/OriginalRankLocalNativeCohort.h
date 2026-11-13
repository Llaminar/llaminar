/**
 * @file OriginalRankLocalNativeCohort.h
 * @brief Pure original-inventory authentication before native component admission.
 *
 * The caller borrows the original rank record and ordered GPU endpoints. CPU
 * affinity, NUMA ownership, ordinal identity and physical UUIDs must agree
 * before native contexts or allocations exist. The complete MPI publication
 * remains intact, including original ranks with no component GPU work.
 */
#pragma once
#include "planning/RankHardwareOwnership.h"
#include <algorithm>
#include <limits>
#include <set>
#include <span>
#include <stdexcept>
#include <string>

namespace llaminar2::test
{
    /**
     * @brief Authenticate exact original rank and physical cohort without discovery.
     * @param inventory Immutable inventory belonging to the original communicator.
     * @param rank Actual current rank in that communicator.
     * @param world_size Original communicator size, independent of GPU cohort width.
     * @param bound_numa Actual caller CPU-affinity observation.
     * @param devices Exact caller-ordered GPU endpoints; permutations are preserved.
     * @return Borrowed original record whose physical resources own admission.
     * @throws std::invalid_argument For mismatched rank, affinity, absent or duplicate endpoints/UUIDs.
     */
    inline const RankInventory &requireOriginalRankLocalNativeCohort(
        const ClusterInventory &inventory, int rank, int world_size, int bound_numa,
        std::span<const DeviceId> devices)
    {
        if (world_size <= 0 || inventory.world_size != world_size ||
            inventory.ranks.size() != static_cast<std::size_t>(world_size) || rank < 0 || rank >= world_size)
            throw std::invalid_argument("Native cohort requires its complete original MPI inventory");
        for (int member = 0; member < world_size; ++member)
            if (inventory.ranks[member].rank != member)
                throw std::invalid_argument("Native cohort original MPI rank identities are incomplete or duplicated");
        const auto &owner = inventory.ranks.at(static_cast<std::size_t>(rank));
        if (owner.cpu.type != DeviceType::CPU || owner.cpu.numa_node < 0 ||
            bound_numa != owner.cpu.numa_node)
            throw std::invalid_argument("Native cohort disagrees with its original rank and CPU NUMA binding");
        if (devices.empty() || devices.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Native cohort requires positive int-representable participant coordinates");
        std::set<std::string> identities;
        for (const auto device : devices)
        {
            if (!device.is_gpu() || std::count(devices.begin(), devices.end(), device) != 1 ||
                std::count_if(owner.gpus.begin(), owner.gpus.end(), [&](const auto &gpu) {
                    return gpu.type == device.type && gpu.local_device_id == device.ordinal;
                }) != 1)
                throw std::invalid_argument("Native cohort has an absent or duplicate original GPU ordinal");
            requireRankLocalGPUOwnership(owner, device, "Original native component cohort");
            const auto found = std::find_if(owner.gpus.begin(), owner.gpus.end(), [&](const auto &gpu) {
                return gpu.type == device.type && gpu.local_device_id == device.ordinal;
            });
            if (found->uuid.empty() || !identities.insert(found->uuid).second)
                throw std::invalid_argument("Native cohort requires distinct observed original GPU UUIDs");
        }
        return owner;
    }
}
