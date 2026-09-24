/**
 * @file LocalTPCollectiveInventory.h
 * @brief Participant-scoped inventory for rank-local collective construction.
 *
 * A LocalTP context already owns the authoritative participant list.  This
 * builder projects that list into the legacy ClusterInventory shape consumed
 * by CollectiveContext without probing CUDA, ROCm, or any other hardware
 * backend.  Avoiding discovery here is an ownership requirement: constructing
 * a CPU+ROCm runner must not create CUDA runtime state merely because CUDA was
 * compiled into the process.
 */

#pragma once

#include "../backends/GlobalDeviceAddress.h"
#include "../execution/mpi_orchestration/DeviceInventory.h"

#include <vector>

namespace llaminar2
{

    /**
     * @brief Build the collective inventory for one LocalTP authority.
     *
     * The returned inventory contains exactly the devices named by
     * @p participants.  It deliberately carries no discovered capacity or
     * capability data because backend routing needs only device type and
     * ordinal; memory planning uses the cluster inventory gathered earlier in
     * the production lifecycle.
     *
     * CPU participants are represented by RankInventory::cpu. CUDA and ROCm
     * participants are represented in RankInventory::gpus in declaration
     * order. Duplicate local device identities and unsupported device types
     * are rejected so the collective router cannot silently operate on a
     * topology different from the LocalTP context.
     *
     * @param participants Authoritative rank-local LocalTP participants.
     * @param mpi_rank MPI rank that owns this LocalTP context.
     * @param mpi_world_size Number of MPI ranks in the enclosing process group.
     * @return A single-rank participant view suitable for CollectiveContext.
     * @throws std::invalid_argument for an empty or invalid participant set,
     *         duplicate devices, or invalid MPI identity.
     */
    [[nodiscard]] ClusterInventory buildLocalTPCollectiveInventory(
        const std::vector<GlobalDeviceAddress> &participants,
        int mpi_rank,
        int mpi_world_size);

} // namespace llaminar2
