/**
 * @file ScopedOpenMPPartitionAffinity.h
 * @brief Scoped placement for independently scheduled control-worker construction.
 *
 * OpenMP binds its initial thread to one place before main(). Ordinary C++
 * children inherit that one-core mask even when MPI admitted a whole socket.
 * Independent GPU-control and HTTP service children need the admitted
 * partition to make progress independently of the inference owner's first
 * place. This scope changes only the thread-construction caller. New children
 * inherit its temporary mask; existing inference, OpenMP, MPI and GPU workers
 * keep their placement. Scope exit restores the caller exactly, without
 * modifying the newly created children or the OpenMP placement policy.
 */
#pragma once

#include <omp.h>
#include <sched.h>

#include <cerrno>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace llaminar2
{
    /** @brief Own temporary control placement within the current rank's admitted CPU partition. */
    class ScopedOpenMPPartitionAffinity
    {
    public:
        /**
         * @brief Admit the caller and future independent control children on its OpenMP partition.
         *
         * Disabled OpenMP binding preserves the caller's native scheduler mask.
         * Enabled binding uses the actual partition's places, not host topology,
         * a guessed NUMA node, or a second rank-placement policy.
         * @throws std::runtime_error for nested use, missing places, or failed publication.
         */
        ScopedOpenMPPartitionAffinity()
        {
            if (omp_get_level() != 0)
                throw std::runtime_error("Independent control placement requires an outer host thread");
            original_ = currentMask();
            const auto desired = serviceMask(original_);
            if (CPU_EQUAL(&original_, &desired))
                return;
            if (sched_setaffinity(0, sizeof(desired), &desired) != 0)
                throw std::system_error(errno, std::generic_category(), "Cannot publish independent control affinity");
            installed_ = true;
            try
            {
                const auto actual = currentMask();
                if (!CPU_EQUAL(&actual, &desired))
                    throw std::runtime_error("Independent control affinity differs from the admitted CPU partition");
            }
            catch (...)
            {
                restore();
                throw;
            }
        }

        /** @brief Restore the construction caller exactly; failure is fatal. Children retain their inherited mask. */
        ~ScopedOpenMPPartitionAffinity() { restore(); }
        ScopedOpenMPPartitionAffinity(const ScopedOpenMPPartitionAffinity &) = delete;
        ScopedOpenMPPartitionAffinity &operator=(const ScopedOpenMPPartitionAffinity &) = delete;

    private:
        /** @return The calling thread's actual scheduler mask; never a process-wide guess. */
        static cpu_set_t currentMask()
        {
            cpu_set_t result;
            CPU_ZERO(&result);
            if (sched_getaffinity(0, sizeof(result), &result) != 0)
                throw std::system_error(errno, std::generic_category(), "Cannot read independent control thread affinity");
            return result;
        }

        /**
         * @brief Resolve the existing OpenMP partition for independent control children.
         * @param unbound Native mask retained when OpenMP binding is explicitly disabled.
         * @return Union of this thread's admitted partition places, or its unbound mask.
         */
        static cpu_set_t serviceMask(const cpu_set_t &unbound)
        {
            if (omp_get_proc_bind() == omp_proc_bind_false)
                return unbound;
            const int count = omp_get_partition_num_places();
            if (count <= 0)
                throw std::runtime_error("Bound independent control caller has no admitted OpenMP places");
            std::vector<int> places(static_cast<std::size_t>(count));
            omp_get_partition_place_nums(places.data());
            cpu_set_t result;
            CPU_ZERO(&result);
            for (int place : places)
            {
                const int width = omp_get_place_num_procs(place);
                if (width <= 0)
                    throw std::runtime_error("Independent control partition contains an empty CPU place");
                std::vector<int> cpus(static_cast<std::size_t>(width));
                omp_get_place_proc_ids(place, cpus.data());
                for (int cpu : cpus)
                {
                    if (cpu < 0 || cpu >= CPU_SETSIZE)
                        throw std::runtime_error("Independent control CPU exceeds the native affinity mask");
                    CPU_SET(cpu, &result);
                }
            }
            return result;
        }

        /** @brief Release temporary placement without allowing an invalid host thread to continue. */
        void restore() noexcept
        {
            if (!installed_)
                return;
            if (sched_setaffinity(0, sizeof(original_), &original_) != 0)
            {
                std::fprintf(stderr, "Fatal: cannot restore independent control caller affinity (errno=%d)\n", errno);
                std::terminate();
            }
            installed_ = false;
        }

        cpu_set_t original_{}; ///< Exact pre-scope placement owned by the calling thread.
        bool installed_ = false; ///< This scope alone may restore its temporary publication.
    };
}
