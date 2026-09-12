/**
 * @file Test__CPUStartupAffinity.cpp
 * @brief Model-free end-to-end CPU selection, MPI self-launch, and OpenMP affinity proof.
 *
 * Exercise the production argument parser and bootstrap instead of supplying a
 * test-owned mpirun command. Every available NUMA endpoint is tested in short
 * and serialized form, together with the serialized unresolved endpoint used
 * by canonical single-CPU generation cells. Each child verifies the complete
 * physical-core team before any model or accelerator allocation is possible.
 */

#include "app/MPIBootstrapPhase.h"
#include "config/OrchestrationConfigParser.h"

#include <mpi.h>
#include <omp.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llaminar2;

namespace
{
    /**
     * @brief Prove an unavailable explicit CPU target fails before self-launch.
     * @return True only for the exact missing-physical-placement diagnostic.
     */
    bool rejectsUnavailableNode()
    {
        char program[] = "cpu-affinity-preflight";
        char flag[] = "--device";
        char selector[] = "localhost:2147483647:cpu:0";
        char *arguments[] = {program, flag, selector};
        OrchestrationConfigParser parser;
        auto config = parser.parseArgs(3, arguments);
        // A dry launch still resolves real physical placement, but can never
        // accidentally start an MPI job if this rejection regresses.
        config.mpi_dry_run = true;
        try
        {
            MPIBootstrapPhase bootstrap;
            (void)bootstrap.execute(config, 3, arguments);
        }
        catch (const std::runtime_error &error)
        {
            return std::string(error.what()).find("has no physical CPU set") !=
                   std::string::npos;
        }
        return false;
    }

    /**
     * @brief Start a fresh production bootstrap for one CPU selector.
     * @param executable Absolute path of this probe.
     * @param selector Exact CLI address to parse in launcher and MPI child.
     * @return True only when bootstrap and every worker's affinity are correct.
     */
    bool checkSelector(const char *executable, const std::string &selector)
    {
        std::cout << "Checking CPU startup: " << selector << std::endl;
        const pid_t child = fork();
        if (child < 0)
            throw std::runtime_error("Cannot fork CPU bootstrap probe");
        if (child == 0)
        {
            execl(executable, executable, "--device", selector.c_str(),
                  "--mpi-procs", "1", nullptr);
            _exit(127);
        }
        int status = 0;
        pid_t waited;
        do { waited = waitpid(child, &status, 0); }
        while (waited < 0 && errno == EINTR);
        return waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }

    /**
     * @brief Run the real parser/bootstrap and check the resulting MPI worker team.
     * @param argc Argument count passed through MPI bootstrap unchanged.
     * @param argv Arguments naming one device and one MPI rank.
     * @return Zero only for one worker per physical core on the selected node.
     */
    int probe(int argc, char **argv)
    {
        OrchestrationConfigParser parser;
        const auto config = parser.parseArgs(argc, argv);
        MPIBootstrapPhase bootstrap;
        const auto launch = bootstrap.execute(config, argc, argv);
        if (launch.action == BootstrapResult::Action::EXIT)
            return launch.exit_code;

        int provided = 0;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        int world_size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        const auto &address = *config.device_for_this_rank;
        const auto topology = MPIBootstrap::detectCPUTopology();
        const auto physical_cpus = MPIBootstrapPhase::parseCpuList(
            MPIBootstrap::getPhysicalCpuSetForNumaNode(address.numa_node));
        const int expected_threads = address.hasValidNuma()
            ? static_cast<int>(physical_cpus.size()) : topology.cores_per_socket;
        std::string details;
        const bool affinity_ok = MPIBootstrapPhase::verifyStartupThreadAffinity(
            address.numa_node, true, details);
        const bool passed = config.device_for_this_rank_numa_explicit == address.hasValidNuma() &&
            world_size == 1 && expected_threads > 0 &&
            omp_get_max_threads() == expected_threads && affinity_ok;
        std::cout << (passed ? "PASS " : "FAIL ") << address.toString()
                  << " workers=" << omp_get_max_threads()
                  << " expected=" << expected_threads << " affinity=" << details
                  << std::endl;
        MPI_Finalize();
        return passed ? 0 : 1;
    }
}

/**
 * @brief Exercise available node selectors, or enter one production self-launch.
 * @return Nonzero if any selector silently changes locality or oversubscribes a core.
 */
int main(int argc, char **argv)
{
    try
    {
        if (argc > 1)
            return probe(argc, argv);
        bool passed = checkSelector(argv[0], "localhost:-1:cpu:0");
        const auto topology = MPIBootstrap::detectCPUTopology();
        for (int node = 0; node < topology.numa_nodes; ++node)
        {
            // Discover nodes from the host rather than assuming two sockets or
            // tying any accelerator vendor to a particular CPU endpoint.
            for (const auto &selector : {
                     "cpu:" + std::to_string(node),
                     "localhost:" + std::to_string(node) + ":cpu:0"})
                passed = checkSelector(argv[0], selector) && passed;
        }
        passed = rejectsUnavailableNode() && passed;
        return passed ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
