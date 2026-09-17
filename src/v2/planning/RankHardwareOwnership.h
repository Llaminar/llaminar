/**
 * @file RankHardwareOwnership.h
 * @brief Pure CPU ownership queries over the canonical rank observation.
 *
 * MPI ordinals describe communicator order, not NUMA locality. Placement and
 * ExpertOverlay use this shared query so reversed communicators and sparse
 * physical NUMA IDs cannot create a second, heuristic hardware inventory.
 */
#pragma once

#include "execution/mpi_orchestration/DeviceInventory.h"
#include <algorithm>

namespace llaminar2
{
    /**
     * @brief Check whether the observed CPU authority covers a requested node.
     * @param rank Immutable hardware/affinity publication for one MPI process.
     * @param numa_node Physical NUMA ID, or -1 for an unqualified CPU request.
     * @return True only for observed ownership, never inferred rank arithmetic.
     *
     * A bound process owns its one observed NUMA endpoint. An explicitly
     * whole-host observation may own several endpoints, but its detailed socket
     * inventory must name them: a node count does not establish their IDs.
     * Missing physical identity is not a whole-host capability advertisement.
     */
    [[nodiscard]] inline bool rankOwnsCPUNode(
        const RankInventory &rank, int numa_node) noexcept
    {
        if (numa_node < -1 || rank.cpu.numa_node < -1)
            return false;
        if (rank.cpu.numa_node >= 0)
            return numa_node == -1 || numa_node == rank.cpu.numa_node;
        return std::any_of(rank.cpu_socket_info.begin(), rank.cpu_socket_info.end(),
            [numa_node](const CPUSocketInfo &socket) {
                return socket.numa_node >= 0 &&
                    (numa_node == -1 || socket.numa_node == numa_node);
            });
    }
}
