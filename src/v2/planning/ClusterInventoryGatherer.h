/**
 * @file ClusterInventoryGatherer.h
 * @brief Canonical startup hardware observation and collective inventory assembly.
 *
 * Placement selectors never enter discovery: requested devices cannot create
 * hardware, overwrite capacities or renumber ordinals. Pure projection accepts
 * a DeviceManager observation; MPI exchanges those records before planning.
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#pragma once

#include "execution/mpi_orchestration/DeviceInventory.h"
#include "interfaces/IMPIContext.h"
#include "backends/HardwareInventory.h"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace llaminar2
{

/** @brief Authenticated MPI identity and CPU memory locality of one observation. */
struct RankHardwareLocation
{
    int rank = 0; ///< Communicator rank, never inferred from a GPU ordinal.
    int node = 0; ///< Physical shared-memory group ID.
    int local_rank = 0; ///< Rank order inside that physical group.
    std::string hostname = "localhost"; ///< Diagnostic label only.
    std::optional<int> cpu_numa_node; ///< Exact owned CPU locality; empty means whole host.
    int cpu_worker_threads = 0; ///< Observed runtime team; zero denotes hardware-only projection.
};

/**
 * @brief Publish every driver-visible accelerator with independently bound CPU ownership.
 * @param hardware Complete, startup-policy-scoped observation owned by DeviceManager.
 * @param location Authenticated rank membership and CPU NUMA affinity.
 * @return Rank observation preserving GPUs on every socket without rediscovery.
 *
 * GPU visibility follows the driver's process namespace, not CPU affinity.
 * Placement chooses the nearest eligible owner later; filtering here would
 * lose a GPU when a hostfile supplies no process on its socket.
 */
RankInventory makeRankInventory(const HardwareInventory &hardware,
                               const RankHardwareLocation &location);

/**
 * @brief Project observed hardware without entering a driver or MPI.
 * @param hardware DeviceManager's immutable startup observation.
 * @param visible_devices Explicit device subset for a scoped collective fixture.
 * @param location Physical membership and CPU endpoint, not a placement request.
 * @return Complete rank record preserving backend ordinals and measured links.
 * @throws std::invalid_argument for an absent locality or malformed observation.
 *
 * This low-level projection serves collective fixtures. Startup discovery uses
 * the complete-observation overload; a locality-filtered execution view is not
 * evidence that other accelerators are absent from the physical host.
 */
RankInventory makeRankInventory(const HardwareInventory &hardware,
                               std::span<const ComputeDevice> visible_devices,
                               const RankHardwareLocation &location);

/**
 * @brief Retain the context-owned inventory, discovering it once if unpublished.
 * @param mpi_ctx Exact discovery context; only nullptr denotes non-MPI discovery.
 * @param hostfile Launch provenance; MPI shared membership owns node identity.
 * @return Shared immutable observation, pointer-identical to the MPI owner's
 *         publication. Explicit local-only callers own their returned observation.
 *
 * Every member enters the first call. Repeated reads neither communicate nor
 * refresh capacity; runtime free-byte admission remains PMA's responsibility.
 * Each rank publishes all startup-policy-permitted GPUs while CPU ownership is
 * affinity-bound. Returning a mutable copy would detach planning evidence from
 * the inventory identity authenticated by the discovery communicator.
 */
std::shared_ptr<const ClusterInventory> gatherClusterInventory(
    const std::shared_ptr<IMPIContext>& mpi_ctx,
    const std::string& hostfile = ""
);

} // namespace llaminar2
