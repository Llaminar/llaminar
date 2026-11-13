/**
 * @file MPISessionSharedDirectory.h
 * @brief Immutable construction-time MPI agreement on original owned SHM roots.
 *
 * This projection uses the existing participant construction communicator. It
 * creates no guardian, file or byte ledger. The process owner remains solely
 * responsible for retaining backing through actual descendant retirement.
 */
#pragma once
#include "utils/MPISessionOwner.h"
#include <mpi.h>

namespace llaminar2
{
    /**
     * @brief Prove physical-node membership and one original shared-directory identity.
     * @param participants Existing startup communicator containing every file consumer.
     * @return Original immutable directory, or unavailable when all ranks are unowned.
     * @throws Collectively for malformed/mixed/mismatched owner facts or nonlocal membership.
     *
     * Call only during communicator/channel construction. Ordinary execution
     * reuses the retained result and performs no MPI handshake or path discovery.
     */
    [[nodiscard]] std::optional<MPISessionOwner::SharedDirectory>
    discoverMPISharedDirectory(MPI_Comm participants);
}
