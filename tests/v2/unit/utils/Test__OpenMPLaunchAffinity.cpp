/**
 * @file Test__OpenMPLaunchAffinity.cpp
 * @brief Launched processes receive the OpenMP-admitted CPU set, not one place.
 *
 * Registered with `OMP_PLACES=cores OMP_PROC_BIND=close`. libgomp then binds
 * every process's initial thread to its first place before `main()`. This
 * executable itself is the MPI session owner's spawned child: before the
 * OpenMPLaunchAffinityScope fix it inherited that single CPU, so its own
 * place list collapsed to one place and every OpenMP team ran on one core.
 *
 * The tests prove (1) the spawned test process still sees one place per
 * admitted core, (2) the scope widens the calling thread to exactly the union
 * of places and restores the bound place afterwards, and (3) a process
 * spawned inside the scope observes the complete admitted set.
 */

#include "utils/OpenMPLaunchAffinity.h"

#include <gtest/gtest.h>
#include <fcntl.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace
{
    /** @return Union of every OpenMP place's CPUs. */
    cpu_set_t placeUnion()
    {
        cpu_set_t result;
        CPU_ZERO(&result);
        for (int place = 0; place < omp_get_num_places(); ++place)
        {
            std::vector<int> ids(static_cast<size_t>(omp_get_place_num_procs(place)));
            omp_get_place_proc_ids(place, ids.data());
            for (const int id : ids)
                CPU_SET(id, &result);
        }
        return result;
    }

    /** @return The calling thread's current affinity. */
    cpu_set_t threadAffinity()
    {
        cpu_set_t result;
        CPU_ZERO(&result);
        EXPECT_EQ(pthread_getaffinity_np(pthread_self(), sizeof(result), &result), 0);
        return result;
    }

    /** @return Number of CPUs in the kernel's affinity list of a child process. */
    int spawnedChildCPUCount()
    {
        char path[] = "/tmp/llaminar-launch-affinity-XXXXXX";
        const int descriptor = mkstemp(path);
        EXPECT_GE(descriptor, 0);
        close(descriptor);
        // The child reports its own kernel affinity count; no shell or helper
        // binary beyond coreutils' nproc, which honours sched_getaffinity.
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, path, O_WRONLY | O_TRUNC, 0600);
        char program[] = "nproc";
        char *argv[] = {program, nullptr};
        pid_t pid = -1;
        // GNU nproc also honours OMP_NUM_THREADS/OMP_THREAD_LIMIT; an empty
        // environment makes it report the inherited kernel affinity alone.
        char *empty_environment[] = {nullptr};
        const int spawned = posix_spawnp(&pid, program, &actions, nullptr, argv, empty_environment);
        posix_spawn_file_actions_destroy(&actions);
        EXPECT_EQ(spawned, 0);
        int status = 0;
        EXPECT_EQ(waitpid(pid, &status, 0), pid);
        EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
        std::ifstream output(path);
        int count = 0;
        output >> count;
        std::remove(path);
        return count;
    }
} // namespace

/** @test The session-owner child keeps one OpenMP place per admitted core. */
TEST(OpenMPLaunchAffinity, SpawnedProcessRetainsAdmittedPlaces)
{
    ASSERT_GT(omp_get_num_places(), 0) << "registration must set OMP_PLACES";
    // Places are cores within the ctest-admitted affinity; a collapsed launch
    // leaves exactly one. Any host this suite admits has at least two CPUs.
    if (sysconf(_SC_NPROCESSORS_ONLN) >= 2)
        EXPECT_GE(omp_get_num_places(), 2);
    int observed = 0;
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        observed = omp_get_num_threads();
    }
    EXPECT_EQ(observed, 2);
}

/** @test The scope widens to the place union and restores the bound place. */
TEST(OpenMPLaunchAffinity, ScopeInstallsUnionThenRestoresBinding)
{
    ASSERT_GT(omp_get_num_places(), 0);
    const cpu_set_t admitted = placeUnion();
    const cpu_set_t bound = threadAffinity();
    {
        const llaminar2::OpenMPLaunchAffinityScope scope;
        const cpu_set_t during = threadAffinity();
        EXPECT_TRUE(CPU_EQUAL(&during, &admitted));
        EXPECT_EQ(scope.widened(), !CPU_EQUAL(&bound, &admitted));
        EXPECT_EQ(spawnedChildCPUCount(), CPU_COUNT(&admitted));
    }
    const cpu_set_t after = threadAffinity();
    EXPECT_TRUE(CPU_EQUAL(&after, &bound));
}
