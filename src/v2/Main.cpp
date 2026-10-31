/**
 * @file Main.cpp
 * @brief Llaminar entrypoint with process-owned MPI runtime files.
 *
 * Enter the shared ancestor owner before AppLifecycle or any runtime state.
 * That owner stays alive after this child becomes mpirun, and cleans only its
 * private roots after all local processes and singleton daemons retire.
 * One lexical GPU scope closes native admission and joins workers before MPI
 * finalization or frontend exit; native cleanup never depends on static order.
 */

#include "utils/MPISessionOwner.h"
#include "app/AppLifecycle.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include "utils/DebugEnv.h"
#include <cstdlib>
#include <unistd.h>
#include <iostream>

int main(int argc, char *argv[])
{
    // Ownership starts before this frontend creates MPI or runtime state.
    if (auto terminal = llaminar2::MPISessionOwner::enterProcess(argc, argv)) return *terminal;
    int result = 0;
    {
        llaminar2::GPUDeviceProcessLifetime gpu_lifetime;
        {
            llaminar2::AppLifecycle app;
            result = app.run(argc, argv);
        }
        // Explicitly retire before the existing _exit policy can bypass C++
        // destructors. MPI-backed modes already retire at mpiShutdown().
        gpu_lifetime.retire();
    }

    // Flush output streams before exit
    std::cout.flush();
    std::cerr.flush();

    if (result == 0 &&
        !llaminar2::debugEnv().runtime_debug.profiler_normal_exit)
    {
        // Skip C++ static destructors via _exit() to avoid RCCL/ROCm CLR bugs.
        // RCCL's internal atexit handler segfaults when cleaning up communicators
        // created via ncclCommInitAll on MI60 GPUs, even after proper
        // ncclCommFinalize + ncclCommDestroy. Our resources are already cleaned
        // up by this point — the OS reclaims the rest.
        _exit(0);
    }
    /* Profilers flush process-owned trace buffers from normal-exit handlers.
     * This opt-in diagnostic path is deliberately never selected implicitly:
     * the ordinary ROCm/RCCL lifecycle keeps the proven `_exit(0)` policy. */
    return result;
}
