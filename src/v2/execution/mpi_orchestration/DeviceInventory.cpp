/**
 * @file DeviceInventory.cpp
 * @brief Physical-node projection of immutable rank hardware observations.
 *
 * Visibility is not exclusive ownership. Union stable UUIDs and NUMA resources
 * without keeping another live memory ledger. This out-of-line implementation
 * keeps map/set instantiations out of the widely included topology contract.
 */
#include "execution/mpi_orchestration/DeviceInventory.h"
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace llaminar2
{
    RankConnectionTopology ClusterInventory::connectionBetweenRanks(int source, int destination) const
    {
        if (world_size <= 0 || ranks.size() != static_cast<size_t>(world_size) ||
            source < 0 || destination < 0 || source >= world_size || destination >= world_size)
            throw std::invalid_argument("Rank connection requires exact cluster membership and present endpoints");
        const auto &a = ranks[source];
        const auto &b = ranks[destination];
        if (a.rank != source || b.rank != destination || a.node_id < 0 || b.node_id < 0)
            throw std::invalid_argument("Rank connection has invalid physical-node identity");
        return {source, destination, a.node_id, b.node_id};
    }

    void ClusterInventory::buildNodeAggregations()
    {
        int max_node = -1;
        for (const auto &r : ranks)
        {
            if (r.node_id < 0)
                throw std::invalid_argument("Inventory rank lacks physical node identity");
            if (r.node_id > max_node)
                max_node = r.node_id;
        }
        if (max_node == std::numeric_limits<int>::max())
            throw std::overflow_error("Inventory node extent overflow");
        std::vector<NodeInventory> rebuilt(static_cast<std::size_t>(max_node + 1));
        // These maps exist only while projecting discovery. They never
        // track reservations, allocations or a runtime free-byte balance.
        using GPUKey = std::pair<DeviceType, std::string>;
        std::vector<std::map<GPUKey, const DeviceInfo *>> gpu_views(rebuilt.size());
        std::vector<std::map<int, std::pair<size_t, int>>> cpu_views(rebuilt.size());
        std::vector<std::set<int>> physical_cores(rebuilt.size());
        std::vector<bool> detailed_cpu(rebuilt.size(), false);
        for (const auto &rank : ranks)
            detailed_cpu[rank.node_id] = detailed_cpu[rank.node_id] ||
                                         !rank.cpu_socket_info.empty();
        for (int n = 0; n <= max_node; ++n)
            rebuilt[n].node_id = n;

        const auto add = [](size_t &total, size_t value)
        {
            if (value > std::numeric_limits<size_t>::max() - total)
                throw std::overflow_error("Inventory capacity summary overflow");
            total += value;
        };
        const auto addCount = [](int &total, size_t value)
        {
            if (value > static_cast<size_t>(std::numeric_limits<int>::max() - total))
                throw std::overflow_error("Inventory device/core count overflow");
            total += static_cast<int>(value);
        };

        for (const auto &r : ranks)
        {
            auto &node = rebuilt[r.node_id];
            if (node.hostname.empty()) node.hostname = r.hostname;
            node.ranks.push_back(r.rank);
            for (const auto &gpu : r.gpus)
            {
                if (!gpu.isGPU())
                    throw std::invalid_argument("Non-GPU in accelerator inventory");
                if (gpu.uuid.empty())
                {
                    // Explicit collective membership has no measured bytes
                    // and cannot be mistaken for admissible GPU capacity.
                    if (gpu.memory_bytes != 0 || gpu.free_memory_bytes != 0)
                        throw std::invalid_argument("Measured GPU inventory requires a driver UUID");
                    addCount(node.total_gpus, 1);
                    continue;
                }
                const auto [it, inserted] = gpu_views[r.node_id].emplace(
                    GPUKey{gpu.type, gpu.uuid}, &gpu);
                if (!inserted)
                {
                    const auto &previous = *it->second;
                    if (previous.memory_bytes != gpu.memory_bytes ||
                        previous.compute_units != gpu.compute_units ||
                        previous.last_level_cache_bytes != gpu.last_level_cache_bytes)
                        throw std::invalid_argument("Conflicting observations of GPU UUID " + gpu.uuid);
                    continue;
                }
                addCount(node.total_gpus, 1);
                add(node.total_gpu_memory, gpu.memory_bytes);
            }

            // A detailed socket observation describes the physical node,
            // even if this particular rank owns only one NUMA endpoint.
            // Union it once instead of summing rank-local memory views.
            if (detailed_cpu[r.node_id])
            {
                for (const auto &socket : r.cpu_socket_info)
                {
                    if (socket.numa_node < 0)
                        throw std::invalid_argument("CPU observation lacks NUMA memory identity");
                    const auto value = std::pair{socket.memory_bytes, 0};
                    const auto [it, inserted] = cpu_views[r.node_id].emplace(socket.numa_node, value);
                    if (!inserted && it->second != value)
                        throw std::invalid_argument("Conflicting observations of NUMA memory capacity");
                    physical_cores[r.node_id].insert(socket.physical_cores.begin(), socket.physical_cores.end());
                }
            }
            else if (r.cpu_memory_bytes != 0 || r.cpu_cores != 0)
            {
                if (r.cpu_cores < 0)
                    throw std::invalid_argument("CPU observation has a negative core count");
                const auto value = std::pair{r.cpu_memory_bytes, r.cpu_cores};
                const auto [it, inserted] = cpu_views[r.node_id].emplace(r.cpu.numa_node, value);
                if (!inserted && it->second != value)
                    throw std::invalid_argument("Conflicting CPU resource observations");
            }
        }

        int rebuilt_gpu_count = 0;
        size_t rebuilt_gpu_bytes = 0, rebuilt_cpu_bytes = 0;
        for (auto &node : rebuilt)
        {
            const auto &cpu = cpu_views[node.node_id];
            if (cpu.contains(-1) && cpu.size() != 1)
                throw std::invalid_argument("Whole-host and NUMA CPU views need physical socket observations");
            for (const auto &[identity, resource] : cpu)
            {
                add(node.total_cpu_memory, resource.first);
                addCount(node.total_cpu_cores, static_cast<size_t>(resource.second));
            }
            if (detailed_cpu[node.node_id])
                addCount(node.total_cpu_cores, physical_cores[node.node_id].size());
            addCount(rebuilt_gpu_count, static_cast<size_t>(node.total_gpus));
            add(rebuilt_gpu_bytes, node.total_gpu_memory);
            add(rebuilt_cpu_bytes, node.total_cpu_memory);
        }
        nodes = std::move(rebuilt);
        node_count = static_cast<int>(nodes.size());
        total_gpus = rebuilt_gpu_count;
        total_gpu_memory = rebuilt_gpu_bytes;
        total_cpu_memory = rebuilt_cpu_bytes;
    }
}
