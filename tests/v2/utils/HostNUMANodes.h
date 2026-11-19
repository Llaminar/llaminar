/**
 * @file HostNUMANodes.h
 * @brief Device-free NUMA topology observation for explicit-placement tests.
 *
 * Placement tests need a node the allocator can actually admit: some hosts
 * (Grace/Neoverse stations) enumerate node ids above every CPU-bearing node
 * as empty memory-only entries, so "highest node id" is not an allocatable
 * placement. The utility therefore returns the highest node that still
 * carries at least one CPU, using the same per-node CPU observation the
 * production NUMAAllocator applies before admitting an explicit node.
 */
#pragma once

#include <numa.h>

namespace llaminar2::test
{
    /**
     * @brief Highest NUMA node id of this host that carries at least one CPU.
     *
     * Purpose: give explicit-placement tests a node the allocator can
     * actually admit. Some hosts enumerate node ids above every CPU-bearing
     * node as empty memory-only entries; such nodes are valid mbind targets
     * for memory but carry no CPUs, and the production allocator rejects
     * CPU placement on them ("Topology exposes no CPU on requested NUMA
     * node"). That rejection is retained: memory-only and nonexistent nodes
     * are never returned by this helper, and callers that need a guaranteed
     * rejection use hostLastNUMANode()+1, which is CPU-less or nonexistent
     * on every topology.
     *
     * @return Highest CPU-bearing node id, or 0 when libnuma reports none
     *     or no node carries a CPU.
     */
    inline int hostLastNUMANode()
    {
        if (numa_available() < 0)
            return 0;
        bitmask *node_cpus = numa_bitmask_alloc(static_cast<unsigned int>(
            numa_max_node() + 1));
        if (!node_cpus)
            return 0;
        int result = 0;
        for (int node = numa_max_node(); node >= 0; --node)
        {
            if (numa_node_to_cpus(node, node_cpus) != 0)
                continue;
            bool has_cpu = false;
            for (unsigned int cpu = 0; cpu < node_cpus->size; ++cpu)
            {
                if (numa_bitmask_isbitset(node_cpus, cpu) != 0)
                {
                    has_cpu = true;
                    break;
                }
            }
            if (has_cpu)
            {
                result = node;
                break;
            }
        }
        numa_bitmask_free(node_cpus);
        return result;
    }
} // namespace llaminar2::test
