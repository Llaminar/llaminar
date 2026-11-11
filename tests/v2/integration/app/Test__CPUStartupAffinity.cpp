/**
 * @file Test__CPUStartupAffinity.cpp
 * @brief Model-free end-to-end CPU selection, MPI self-launch, and OpenMP affinity proof.
 *
 * Exercise the production argument parser and bootstrap instead of supplying a
 * test-owned mpirun command. Every available NUMA endpoint is tested in short
 * and serialized form, together with the serialized unresolved endpoint used
 * by canonical single-CPU generation cells. Each child verifies the complete
 * physical-core team before any model or accelerator allocation is possible,
 * including that every close team occupies the fastest physical cores first.
 * A separate placement contract starts the caller on a different admitted core
 * before cluster configuration, proving temporary admission cannot overwrite
 * the permanent master placement when its scope retires.
 *
 * The custom entry owns one lexical native GPU lifetime without probing a
 * device. Native workers/caches retire before process exit and MPI finalization.
 */

#include "backends/GPUDeviceProcessLifetime.h"
#include "utils/MPISessionOwner.h"
#include "app/MPIBootstrapPhase.h"
#include "config/OrchestrationConfigParser.h"
#include "utils/ScopedOpenMPPartitionAffinity.h"

#include <mpi.h>
#include <omp.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <sched.h>
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
        catch (const std::invalid_argument &error)
        {
            return std::string(error.what()) ==
                   "MPI placement cannot select an unknown/duplicate observed NUMA node";
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
     * @brief Prove every close team occupies the rank's fastest physical cores.
     * @param details Receives the first mismatching (team width, worker, CPU).
     * @return True when, for the full team and every narrower close team, each
     *         worker t runs inside capacity-ordered place t of the rank mask.
     * @details On GB10 this is the X925-first contract that serial master work,
     * static partitions and a fastest-class team depend on. Hosts without
     * published capacities check libgomp's own CPU-id core order instead.
     */
    bool workersFollowCapacityOrder(std::string &details)
    {
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        if (sched_getaffinity(0, sizeof(affinity), &affinity) != 0)
            return false;
        std::vector<int> allowed;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &affinity)) allowed.push_back(cpu);
        // The master's own mask is its place after binding; collect the rank
        // mask from every worker's place instead of from the bound master.
        const int team = omp_get_max_threads();
        std::vector<std::vector<int>> masks(team);
#pragma omp parallel num_threads(team)
        {
            cpu_set_t own;
            CPU_ZERO(&own);
            if (sched_getaffinity(0, sizeof(own), &own) == 0)
                for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
                    if (CPU_ISSET(cpu, &own)) masks[omp_get_thread_num()].push_back(cpu);
        }
        for (const auto &mask : masks) allowed.insert(allowed.end(), mask.begin(), mask.end());
        std::sort(allowed.begin(), allowed.end());
        allowed.erase(std::unique(allowed.begin(), allowed.end()), allowed.end());
        const auto places = MPIBootstrapPhase::capacityOrderedCorePlaces(allowed);
        std::vector<int> widths{team, (team + 1) / 2, 1};
        widths.erase(std::unique(widths.begin(), widths.end()), widths.end());
        for (const int width : widths)
        {
            std::vector<int> observed(width, -1);
#pragma omp parallel num_threads(width)
            observed[omp_get_thread_num()] = sched_getcpu();
            for (int worker = 0; worker < width; ++worker)
            {
                const auto &place = places[std::size_t(worker) % places.size()];
                if (std::find(place.begin(), place.end(), observed[worker]) == place.end())
                {
                    details += " capacity-order mismatch: width=" + std::to_string(width) +
                        " worker=" + std::to_string(worker) + " cpu=" + std::to_string(observed[worker]);
                    return false;
                }
            }
        }
        details += " capacity-ordered";
        return true;
    }

    /**
     * @brief Prove rank admission preserves the permanent team placement after scope retirement.
     * @param admission This caller's native place publication before worker admission.
     * @return Zero only when every full/narrow/repeated team retains its original admitted places.
     * @details Starting the caller on the last capacity-ordered place makes restoration of a
     *          stale caller mask distinguishable even on homogeneous physical cores. No model,
     *          MPI launch or accelerator is needed; the production cluster configurator owns
     *          the final placement and the test never repairs its result.
     */
    int checkClusterMasterPlacement(const NativeOpenMPCorePlaces &admission)
    {
        const auto &places=admission.places();
        if (places.size()<2)
            throw std::runtime_error("Cluster master placement contract requires two physical core places");
        const int physical=static_cast<int>(places.size());
        for (const int requested:{0,physical,(physical+1)/2,1})
        {
            cpu_set_t initial;
            CPU_ZERO(&initial);
            for (const int cpu:places.back())CPU_SET(cpu,&initial);
            if (sched_setaffinity(0,sizeof(initial),&initial)!=0)
                throw std::runtime_error("Cannot install the test caller's admitted starting place");
            OrchestrationConfig config;
            config.n_threads=requested;
            MPIBootstrapPhase::configureClusterRankThreads(config,admission);
            const int team=requested>0?requested:physical;
            if (omp_get_max_threads()!=team)
                throw std::runtime_error("Cluster admission changed the original physical worker budget");
            int phase=0;
            for (const int width:{team,(team+1)/2,1,team})
            {
                ++phase;
                std::vector<int> observed(static_cast<size_t>(width),-1);
                std::vector<cpu_set_t> masks(static_cast<size_t>(width));
                std::vector<int> mask_errors(static_cast<size_t>(width));
#pragma omp parallel num_threads(width)
                {
                    const int worker=omp_get_thread_num();
                    observed[worker]=sched_getcpu();
                    CPU_ZERO(&masks[worker]);
                    mask_errors[worker]=sched_getaffinity(0,sizeof(cpu_set_t),&masks[worker]);
                }
                for (int worker=0;worker<width;++worker)
                {
                    const auto &expected=places[static_cast<size_t>(worker)];
                    cpu_set_t expected_mask;
                    CPU_ZERO(&expected_mask);
                    for(const int cpu:expected)CPU_SET(cpu,&expected_mask);
                    if (mask_errors[worker]!=0 || !CPU_EQUAL(&masks[worker],&expected_mask) ||
                        std::find(expected.begin(),expected.end(),observed[worker])==expected.end())
                    {
                        std::cerr<<"Cluster permanent placement changed: requested="<<requested
                                 <<" phase="<<phase<<" width="<<width<<" worker="<<worker<<" cpu="<<observed[worker]
                                 <<" expected_first_cpu="<<expected.front()<<" observed=";
                        for (const int cpu:observed)std::cerr<<cpu<<',';
                        std::cerr<<std::endl;
                        return 1;
                    }
                }
            }
        }
        std::cout<<"PASS cluster permanent master and repeated team placement"<<std::endl;
        return 0;
    }

    /**
     * @brief Exercise original guardian loans after actual native placement entry.
     * @return Zero only when the same native process can use its authenticated shared backing.
     */
    int checkNativeStartupOwnership()
    {
        (void)NativeCPUPlacement::admission();
        auto directory=MPISessionOwner::sharedDirectory();
        if(!directory)throw std::runtime_error("Native startup lost its original shared directory loan");
        const auto identity=directory->identity();
        auto file=directory->createExclusive("native-core-entry-proof");
        if(file.descriptor()<0)throw std::runtime_error("Native startup cannot use its original backing loan");
        file.unlink();
        std::cout<<"PASS native_entry_pid="<<::getpid()
                 <<" shared_device="<<identity.device<<" shared_inode="<<identity.inode<<std::endl;
        return 0;
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
        const bool order_ok = workersFollowCapacityOrder(details);
        const bool passed = config.device_for_this_rank_numa_explicit == address.hasValidNuma() &&
            world_size == 1 && expected_threads > 0 &&
            omp_get_max_threads() == expected_threads && affinity_ok && order_ok;
        std::cout << (passed ? "PASS " : "FAIL ") << address.toString()
                  << " workers=" << omp_get_max_threads()
                  << " expected=" << expected_threads << " affinity=" << details
                  << std::endl;
        llaminar2::GPUDeviceProcessLifetime::retireActive();
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
    // Ownership starts before this frontend creates MPI or runtime state.
    if (auto terminal = llaminar2::MPISessionOwner::enterProcess(argc, argv)) return *terminal;
    llaminar2::GPUDeviceProcessLifetime gpu_process_lifetime;

    try
    {
        if (argc==2 && std::string(argv[1])=="--native-startup-ownership-contract")
            return checkNativeStartupOwnership();
        if (argc==2 && std::string(argv[1])=="--cluster-master-placement-contract")
        {
            return checkClusterMasterPlacement(NativeCPUPlacement::admission());
        }
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
