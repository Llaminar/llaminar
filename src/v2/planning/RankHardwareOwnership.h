/**
 * @file RankHardwareOwnership.h
 * @brief Pure CPU and GPU ownership queries over the canonical rank observation.
 *
 * MPI ordinals describe communicator order, not NUMA locality. Placement and
 * ExpertOverlay use this shared query so reversed communicators and sparse
 * physical NUMA IDs cannot create a second, heuristic hardware inventory.
 */
#pragma once

#include "execution/mpi_orchestration/DeviceInventory.h"
#include "backends/DeviceId.h"
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

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
    /**
     * @brief Admit execution ownership only on the GPU's observed NUMA node.
     * @param rank Canonical process CPU-affinity observation, not its MPI ordinal.
     * @param gpu Canonical GPU observation in that process's visible namespace.
     * @return True only for a GPU and two known, equal physical NUMA identities.
     * Full-host visibility never grants remote or unqualified execution ownership.
     */
    [[nodiscard]] inline bool rankOwnsGPUDevice(
        const RankInventory &rank, const DeviceInfo &gpu) noexcept
    {
        return (gpu.type == DeviceType::CUDA || gpu.type == DeviceType::ROCm) &&
            rank.cpu.numa_node >= 0 && gpu.numa_node >= 0 &&
            rank.cpu.numa_node == gpu.numa_node;
    }

    /**
     * @brief Reject a visible GPU whose execution owner is not demonstrably local.
     * @param rank Original owner observation; no binding or inventory is changed.
     * @param gpu Original visible GPU observation.
     * @param context Calling domain/plan identity for an actionable startup error.
     * @throws std::invalid_argument For remote or unknown physical locality.
     */
    inline void requireRankLocalGPUOwnership(const RankInventory &rank,
        const DeviceInfo &gpu, const std::string &context)
    {
        if (rankOwnsGPUDevice(rank, gpu)) return;
        std::ostringstream error;
        error << context << ": GPU ownership requires matching observed NUMA locality; MPI rank "
              << rank.rank << " CPU NUMA=" << rank.cpu.numa_node << " cannot own "
              << deviceTypeToString(gpu.type) << ":" << gpu.local_device_id
              << " GPU NUMA=" << gpu.numa_node
              << ". Assign this GPU to a rank bound to its NUMA node; full-host discovery is visibility only";
        throw std::invalid_argument(error.str());
    }

    /**
     * @brief Authenticate one selected GPU role before native materialization.
     * @param rank Canonical process observation, including every visible GPU.
     * @param device Actual compute, storage, coordinator or follower endpoint.
     * @param context Calling role identity for startup diagnostics.
     * @throws std::invalid_argument For absent endpoints or unproven NUMA locality.
     * Lookup does not narrow discovery or change CPU execution placement.
     */
    inline void requireRankLocalGPUOwnership(const RankInventory &rank,
        DeviceId device, const std::string &context)
    {
        const auto found = std::find_if(rank.gpus.begin(), rank.gpus.end(),
            [&](const DeviceInfo &gpu) {
                return gpu.type == device.type && gpu.local_device_id == device.ordinal;
            });
        if (!device.is_gpu() || found == rank.gpus.end())
            throw std::invalid_argument(context + ": selected GPU is absent from the original rank inventory: " + device.toString());
        requireRankLocalGPUOwnership(rank, *found, context);
    }

}
