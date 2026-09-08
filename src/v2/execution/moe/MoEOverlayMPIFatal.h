#pragma once

/**
 * @file MoEOverlayMPIFatal.h
 * @brief Unbuffered diagnostics for irrecoverable ExpertOverlay MPI failures.
 *
 * ExpertOverlay maintenance uses private communicators so its asynchronous
 * control traffic cannot collide with inference collectives.  Once an MPI
 * request fails or exceeds its deadline, request ownership and collective
 * order are no longer knowable.  The only safe response is to abort that
 * communicator.  MPI implementations may terminate peers before ordinary
 * logger buffers are drained, however, so every such abort passes through the
 * helper below and emits one complete diagnostic directly to `stderr` first.
 */

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace llaminar2
{
    /**
     * @brief Report an irrecoverable maintenance-lane failure and abort MPI.
     *
     * The explicit lane name distinguishes independently duplicated
     * communicators, while `detail` carries the active protocol phase or MPI
     * error.  `fflush` is intentional: `MPI_Abort` may kill every process
     * immediately, before the normal asynchronous logger writes its buffers.
     * If a non-conforming MPI implementation returns, the process is aborted
     * directly because returning to inference would reuse an indeterminate
     * message stream.
     *
     * @param communicator Private communicator whose ordering is compromised.
     * @param world_rank Rank reporting the failure in the parent MPI world.
     * @param lane Stable subsystem/lane identifier.
     * @param detail Exact operation, state, and reason for the failure.
     */
    [[noreturn]] inline void abortMoEOverlayMPI(
        MPI_Comm communicator,
        int world_rank,
        const char *lane,
        const std::string &detail) noexcept
    {
        std::fprintf(
            stderr,
            "[ExpertOverlay][FATAL][MPI] rank=%d lane=%s detail=%s\n",
            world_rank,
            lane ? lane : "unknown",
            detail.c_str());
        std::fflush(stderr);

        const int abort_status = MPI_Abort(communicator, EXIT_FAILURE);
        std::fprintf(
            stderr,
            "[ExpertOverlay][FATAL][MPI] rank=%d lane=%s MPI_Abort unexpectedly returned status=%d\n",
            world_rank,
            lane ? lane : "unknown",
            abort_status);
        std::fflush(stderr);
        std::abort();
    }
} // namespace llaminar2
