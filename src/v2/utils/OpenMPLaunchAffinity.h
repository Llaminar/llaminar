/**
 * @file OpenMPLaunchAffinity.h
 * @brief Give spawned/exec'd processes the admitted CPU set, not the CPU that
 *        the OpenMP runtime pinned this launching thread to.
 *
 * When `OMP_PLACES` or `OMP_PROC_BIND` is present at process start, libgomp
 * binds the initial thread to the first place before `main()` runs. Linux
 * children inherit the *calling thread's* affinity, so a process that then
 * spawns its MPI session child (MPISessionOwner) or execs `mpirun`
 * (MPIBootstrap) would hand every descendant one CPU: its OpenMP places
 * collapse to `{0}` and the whole team runs on one core.
 *
 * libgomp derives its place list from the original process affinity and only
 * admits CPUs inside that mask, so the union of every place is exactly the
 * admitted CPU set (restricted by the place granularity). This owner widens
 * the launching thread to that union for the duration of one spawn/exec and
 * restores the thread afterwards. Without bound places (no OpenMP runtime in
 * the process, or no places) it does nothing: the current affinity is then
 * already the original one.
 *
 * The OpenMP entry points are weak references: the standalone MPI session
 * launcher deliberately links no OpenMP runtime and therefore never narrows
 * its own affinity. Any failure to install the admitted set is fatal; a
 * descendant must never silently inherit a one-core team.
 */

#pragma once

#include <omp.h>
#include <pthread.h>
#include <sched.h>

#include <stdexcept>

// Weak references keep the declarations' own exception specifications; an
// executable without an OpenMP runtime resolves them to null.
#pragma weak omp_get_num_places
#pragma weak omp_get_place_num_procs
#pragma weak omp_get_place_proc_ids

namespace llaminar2
{
    /** @brief RAII: the calling thread holds the OpenMP-admitted CPU set. */
    class OpenMPLaunchAffinityScope
    {
    public:
        /**
         * @brief Widen the calling thread to the union of all OpenMP places.
         * @throws std::runtime_error When the admitted set cannot be read or installed.
         */
        OpenMPLaunchAffinityScope()
        {
            if (&omp_get_num_places == nullptr || &omp_get_place_num_procs == nullptr ||
                &omp_get_place_proc_ids == nullptr)
                return;
            const int places = omp_get_num_places();
            if (places <= 0)
                return;
            cpu_set_t admitted;
            CPU_ZERO(&admitted);
            int ids[CPU_SETSIZE];
            for (int place = 0; place < places; ++place)
            {
                const int count = omp_get_place_num_procs(place);
                if (count <= 0 || count > CPU_SETSIZE)
                    throw std::runtime_error("OpenMP place has no admitted CPU for a launched child");
                omp_get_place_proc_ids(place, ids);
                for (int index = 0; index < count; ++index)
                {
                    if (ids[index] < 0 || ids[index] >= CPU_SETSIZE)
                        throw std::runtime_error("OpenMP place names an invalid CPU for a launched child");
                    CPU_SET(ids[index], &admitted);
                }
            }
            if (pthread_getaffinity_np(pthread_self(), sizeof(previous_), &previous_) != 0)
                throw std::runtime_error("Cannot read the launching thread's CPU affinity");
            if (CPU_EQUAL(&previous_, &admitted))
                return;
            if (pthread_setaffinity_np(pthread_self(), sizeof(admitted), &admitted) != 0)
                throw std::runtime_error("Cannot give a launched child the OpenMP-admitted CPU set");
            armed_ = true;
        }

        /** @brief Restore the launching thread's own OpenMP binding. */
        ~OpenMPLaunchAffinityScope()
        {
            if (armed_)
                (void)pthread_setaffinity_np(pthread_self(), sizeof(previous_), &previous_);
        }

        OpenMPLaunchAffinityScope(const OpenMPLaunchAffinityScope &) = delete;
        OpenMPLaunchAffinityScope &operator=(const OpenMPLaunchAffinityScope &) = delete;

        /** @return Whether the calling thread was widened for the launch. */
        bool widened() const noexcept { return armed_; }

    private:
        cpu_set_t previous_{}; ///< Thread affinity before the launch.
        bool armed_ = false;   ///< True only after a successful widening.
    };
} // namespace llaminar2
