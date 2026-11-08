/**
 * @file MPIProcessAffinityProbe.cpp
 * @brief Replay real process entry and child placement without MPI or accelerators.
 *
 * An external observer supplies the original scheduler domain before OpenMP
 * binds this executable's initial thread. The normal owner then reexecutes this
 * exact executable. Actual managed workers publish their native masks, while
 * the waiting frontend proves its original single-place mask was restored.
 */
#include "utils/MPISessionOwner.h"
#include <omp.h>
#include <sched.h>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    /** @return Actual calling-thread scheduler mask, rejecting incomplete observations. */
    cpu_set_t observedMask()
    {
        cpu_set_t result;
        CPU_ZERO(&result);
        if (::sched_getaffinity(0, sizeof(result), &result) != 0)
            throw std::runtime_error("Cannot observe actual process-entry CPU mask");
        return result;
    }

    /** @brief Publish one exact native mask as a bounded CPU-list observation. */
    void publishMask(std::ostream &stream, const cpu_set_t &mask)
    {
        bool first = true;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &mask))
            {
                if (!first) stream << ',';
                stream << cpu;
                first = false;
            }
    }
}

/** @brief Enter the actual process owner, then publish every real worker's received placement. */
int main(int argc, char **argv)
{
    try
    {
        if (argc != 3) throw std::invalid_argument("Usage: process-affinity-probe OBSERVATION WORKERS");
        const int workers = std::stoi(argv[2]);
        if (workers < 1) throw std::invalid_argument("Expected worker count must be positive");
        const auto original = observedMask();
        if (const auto terminal = llaminar2::MPISessionOwner::enterProcess(argc, argv))
        {
            const auto restored = observedMask();
            std::ofstream parent(std::string(argv[1]) + ".parent");
            parent << "original="; publishMask(parent, original);
            parent << "\nrestored="; publishMask(parent, restored); parent << '\n';
            if (!parent || !CPU_EQUAL(&original, &restored))
                throw std::runtime_error("Process ownership changed its waiting caller's original affinity");
            return *terminal;
        }
        llaminar2::MPISessionOwner::requireProcessOwner();
        std::vector<cpu_set_t> masks(static_cast<std::size_t>(workers));
        std::vector<int> cpus(static_cast<std::size_t>(workers), -1);
        std::vector<int> errors(static_cast<std::size_t>(workers), 0);
        int received = 0;
#pragma omp parallel
        {
            const int index = omp_get_thread_num();
#pragma omp single
            received = omp_get_num_threads();
            if (index < workers)
            {
                CPU_ZERO(&masks[index]);
                errors[index] = ::sched_getaffinity(0, sizeof(cpu_set_t), &masks[index]);
                cpus[index] = ::sched_getcpu();
            }
        }
        if (received != workers) throw std::runtime_error("Actual OpenMP team differs from requested workers");
        std::ofstream output(argv[1]);
        output << "received=" << received << '\n';
        for (int worker = 0; worker < workers; ++worker)
        {
            if (errors[worker] != 0 || cpus[worker] < 0)
                throw std::runtime_error("Actual worker affinity observation failed");
            output << worker << ' ' << cpus[worker] << ' ';
            publishMask(output, masks[worker]); output << '\n';
        }
        if (!output) throw std::runtime_error("Cannot publish actual managed-worker observation");
        return 0;
    }
    catch (const std::exception &failure)
    {
        std::fprintf(stderr, "%s\n", failure.what());
        return 125;
    }
}
