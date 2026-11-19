/**
 * @file HostNUMANodes.h
 * @brief Host NUMA node identities for tests that bind real CPU memory.
 *
 * Tests that exercise an explicit, non-default NUMA node must name a node that
 * exists on the executing host. Single-node hosts (GB10, desktop x86) expose
 * one node. The last online node
 * keeps the original dual-socket behaviour while remaining valid everywhere.
 */
#pragma once

#include <numa.h>
#include <stdexcept>

namespace llaminar2::test
{
    /**
     * @return Highest online NUMA node for a test using a physical placement.
     * @throws std::runtime_error when the host cannot supply an explicit node.
     */
    inline int hostLastNUMANode()
    {
        if (numa_available() < 0)
            throw std::runtime_error("Explicit test placement requires an online NUMA node");
        return numa_max_node();
    }
} // namespace llaminar2::test
