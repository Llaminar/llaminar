/**
 * @file HostNUMANodes.h
 * @brief Host NUMA node identities for tests that bind real CPU memory.
 *
 * Tests that exercise an explicit, non-default NUMA node must name a node that
 * exists on the executing host. Single-node hosts (GB10, desktop x86) expose
 * only node 0; dual-socket servers expose node 1 as well. The last online node
 * keeps the original dual-socket behaviour while remaining valid everywhere.
 */
#pragma once

#include <numa.h>

namespace llaminar2::test
{
    /** @return Highest NUMA node id of this host, or 0 when libnuma reports none. */
    inline int hostLastNUMANode()
    {
        return numa_available() < 0 ? 0 : numa_max_node();
    }
} // namespace llaminar2::test
