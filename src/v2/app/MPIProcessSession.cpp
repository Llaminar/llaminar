/**
 * @file MPIProcessSession.cpp
 * @brief One process initialization/finalization scope shared by every frontend.
 *
 * Initial thread support is agreed before creating any communicator owners.
 * Thereafter normal C++ scope order retires request handlers, runners and MPI
 * contexts before the existing infrastructure shutdown boundary is entered.
 * No idle-rank loop, model-state mirror or inference-time collective is added.
 */
#include "app/MPIProcessSession.h"
#include "app/MPIShutdown.h"
#include <mpi.h>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    MPIProcessSession MPIProcessSession::initialize(int &argc, char **&argv)
    {
        int initialized = 0, finalized = 0;
        if (MPI_Initialized(&initialized) != MPI_SUCCESS || MPI_Finalized(&finalized) != MPI_SUCCESS)
            throw std::runtime_error("Cannot inspect process MPI lifecycle");
        if (initialized || finalized)
            throw std::logic_error("MPI process initialization requires an unowned, never-finalized runtime");

        int provided = 0;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS)
            throw std::runtime_error("Failed to initialize process MPI runtime");
        MPIProcessSession session(Ownership::Owned);
        int compatible = provided >= MPI_THREAD_MULTIPLE;
        // All ranks must agree before any can enter discovery. On rejection
        // every armed scope unwinds together, rather than stranding a peer.
        if (MPI_Allreduce(MPI_IN_PLACE, &compatible, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) != MPI_SUCCESS)
            throw std::runtime_error("Cannot authenticate process MPI thread support");
        if (!compatible)
            throw std::runtime_error("Llaminar requires MPI_THREAD_MULTIPLE on every process");
        return session;
    }

    MPIProcessSession::MPIProcessSession(MPIProcessSession &&other) noexcept
        : ownership_(std::exchange(other.ownership_, Ownership::Unowned)) {}

    MPIProcessSession::~MPIProcessSession()
    {
        if (ownsMPI()) mpiShutdown();
    }
}
