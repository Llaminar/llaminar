/**
 * @file ServerDiscoveryMembership.h
 * @brief Describe the original communicator at the existing serving admission boundary.
 */
#pragma once

#include "interfaces/IMPIContext.h"
#include "planning/ExecutionRankMembership.h"

#include <sstream>
#include <string>
#include <unistd.h>

namespace llaminar2
{
    /**
     * @brief Format a native PID witness after discovery membership consensus.
     * @param discovery Original communicator, before rank rebinding.
     * @param selection The same ordered cohort authenticated by selectRanks.
     * @return A single machine-readable startup record. No MPI or device call is performed.
     */
    inline std::string serverDiscoveryMembership(const IMPIContext &discovery,
                                                 const ExecutionRankSelection &selection)
    {
        std::ostringstream record;
        record << "[ServerDiscoveryMembership] schema=1 pid=" << ::getpid()
               << " discovery_rank=" << discovery.rank() << " discovery_size=" << discovery.world_size()
               << " selected=";
        const auto &selected = selection.discoveryRanks();
        for (std::size_t i = 0; i < selected.size(); ++i)
        {
            if (i) record << ',';
            record << selected[i];
        }
        return record.str();
    }
}
