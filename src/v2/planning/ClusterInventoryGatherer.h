/**
 * @file ClusterInventoryGatherer.h
 * @brief Free function to gather cluster-wide device inventory from all MPI ranks
 *
 * Extracted from OrchestrationRunner::gatherClusterInventory() so that
 * the inventory gathering logic can be reused by MemoryPlanner and other
 * planning components without depending on OrchestrationRunner.
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#pragma once

#include "execution/mpi_orchestration/DeviceInventory.h"
#include "interfaces/IMPIContext.h"
#include "backends/GlobalDeviceAddress.h"

#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

/// Gather cluster-wide device inventory from all MPI ranks.
/// Single-rank path: returns every device on the local host (no MPI), because
/// one process owns every NUMA domain. Multi-rank discovery retains rank-local
/// NUMA filtering so hardware binding can select the closest owner rank.
/// Multi-rank path: MPI_Allgatherv exchange of serialized RankInventory.
///
/// @param mpi_ctx MPI context (nullptr or world_size==1 → local only)
/// @param explicit_tp_devices If non-empty, overrides detected GPUs with this list
/// @param hostfile Optional hostfile for deterministic node-ID assignment
/// @return Complete ClusterInventory visible to all ranks
ClusterInventory gatherClusterInventory(
    const std::shared_ptr<IMPIContext>& mpi_ctx,
    const std::vector<GlobalDeviceAddress>& explicit_tp_devices = {},
    const std::string& hostfile = ""
);

} // namespace llaminar2
