/**
 * @file Test__ClusterInventoryBootstrap.cpp
 * @brief Model-free production hostfile bootstrap and inventory exchange proof.
 *
 * The parent creates a local hostfile and calls the real parser/bootstrap.
 * MPI must use every declared slot without inheriting a launcher-local CPU
 * set. Children prove their full physical worker binding, then collectively
 * gather the canonical CPU inventory. A bound single rank retains its measured
 * NUMA ownership, just like a distributed rank; rank count is not locality.
 * Repeated rank-zero reads and topology
 * construction must share that exact publication without another exchange.
 * Server evidence must project those same physical groups without a new probe
 * or collective, including when only the request authority reads the snapshot.
 * Production CPU domain/overlay binding must also preserve physical ownership
 * when communicator rank order is reversed. Pipeline activation edges and
 * endpoint actions retain that same immutable membership projection.
 * No model or accelerator is opened.
 * This proves a real multi-process node, not a remote-network deployment.
 * A separate unbound-runtime lifecycle check calls actual rank-worker admission without
 * MPI or devices. Without original-entry native core admission, admission is
 * rejected before any binding: the exact caller/child masks, unbound policy and
 * unpinned workers survive automatic, partial and repeated requests, as well as
 * nested-call rejection.
 *
 * The custom entry owns one lexical native GPU lifetime without probing a
 * device. Native workers/caches retire before process exit and MPI finalization.
 */
#include <algorithm>
#include <map>
#include <numa.h>
#include "backends/GPUDeviceProcessLifetime.h"
#include "fixtures/execution/RetainedHostServiceReplayAccess.h"
#include "execution/local_execution/graph/RetainedParentTicketServiceWorker.h"
#include "utils/ScopedOpenMPPartitionAffinity.h"
#include "utils/ScopedCallerThreadAffinity.h"
#include "utils/MPISessionOwner.h"
#include "utils/NativeCPUPlacement.h"
#include "app/MPIBootstrapPhase.h"
#include "app/RuntimeInitPhase.h"
#include "app/modes/ServerRankMembership.h"
#include "utils/NUMATopology.h"
#include "config/OrchestrationConfigParser.h"
#include "planning/ClusterInventoryGatherer.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "execution/global_pp/GlobalPPRankPlanBuilder.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "utils/MPIContext.h"
#include "utils/MPITopology.h"
#include <mpi.h>
#include <omp.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <memory>
#include <atomic>
#include <set>
#include <map>
#include <sched.h>
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace llaminar2;

namespace
{
    /** @return The calling OS thread's exact native allowed mask, or fail before admission. */
    cpu_set_t callerMask()
    {
        cpu_set_t result{};
        if (sched_getaffinity(0, sizeof(result), &result) != 0)
            throw std::runtime_error("Cannot observe bootstrap lifecycle caller affinity");
        return result;
    }

    /**
     * @brief Check that an unbound runtime cannot enter rank-worker binding.
     * @return Zero only when every rejected request leaves caller, child and
     *         worker masks exactly at the original rank partition.
     * @throws std::runtime_error For missing native geometry or any failed invariant.
     *
     * Rank-worker configuration consumes original-entry native core admission.
     * An unbound runtime (OMP_PROC_BIND=false) owns no such admission, so
     * production rejects it before team binding could pin the master to a
     * singleton that native control children would inherit. Two distinct
     * physical cores make any leaked singleton observable without depending on
     * CPU IDs, NUMA labels, MPI ranks or accelerators. Only this test caller is
     * narrowed; its original mask is restored on exit.
     */
    int checkBootstrapCallerAffinity()
    {
        const ScopedCallerThreadAffinity restore_test_caller;
        if (omp_get_proc_bind() != omp_proc_bind_false)
            throw std::runtime_error("Bootstrap caller lifecycle requires pre-exec OMP_PROC_BIND=false");
        if (NativeCPUPlacement::hasCoreAdmission())
            throw std::runtime_error("Unbound runtime unexpectedly owns native core admission");
        const auto original = callerMask();
        std::map<int, int> physical;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &original))
                physical.try_emplace(MPIBootstrapPhase::physicalRepresentativeForCpu(cpu), cpu);
        if (physical.size() < 2 || physical.contains(-1))
            throw std::runtime_error("Bootstrap caller lifecycle requires two admitted physical cores");
        cpu_set_t admitted{};
        int selected = 0;
        for (const auto &[representative, cpu] : physical)
        {
            CPU_SET(cpu, &admitted);
            if (++selected == 2) break;
        }
        if (sched_setaffinity(0, sizeof(admitted), &admitted) != 0)
            throw std::runtime_error("Cannot admit bootstrap lifecycle caller cores");
        {
            const ScopedOpenMPPartitionAffinity partition;
            const auto unchanged = callerMask();
            if (!CPU_EQUAL(&admitted, &unchanged))
                throw std::runtime_error("Unbound partition unexpectedly changed caller mask");
        }
        const int team_before = omp_get_max_threads();

        // Zero is production auto sizing; the remaining requests cover full,
        // partial and repeated teams. Each must be rejected before binding.
        for (const int requested : {0, 2, 1, 2, 0})
        {
            OrchestrationConfig config;
            config.n_threads = requested;
            bool rejected = false;
            try { MPIBootstrapPhase::configureClusterRankThreads(config, NativeCPUPlacement::admission()); }
            catch (const std::runtime_error &error)
            {
                rejected = std::string(error.what()).find("native CPU core admission") != std::string::npos;
            }
            if (!rejected)
                throw std::runtime_error("Unbound runtime entered rank-worker binding without native core admission");
            const auto retained = callerMask();
            if (!CPU_EQUAL(&admitted, &retained))
                throw std::runtime_error("Rejected bootstrap changed its caller's rank mask");
            bool child_inherited = false;
            std::thread child([&] {
                cpu_set_t inherited{};
                child_inherited = sched_getaffinity(0, sizeof(inherited), &inherited) == 0 &&
                    CPU_EQUAL(&admitted, &inherited);
            });
            child.join();
            if (!child_inherited)
                throw std::runtime_error("Native child lost the original rank mask");
            if (omp_get_max_threads() != team_before || omp_get_proc_bind() != omp_proc_bind_false)
                throw std::runtime_error("Rejected bootstrap changed the runtime worker policy");
            const int width = 2;
            std::vector<cpu_set_t> masks(static_cast<std::size_t>(width));
            int actual = 0, failed = 0;
#pragma omp parallel num_threads(width) reduction(| : failed)
            {
#pragma omp single
                actual = omp_get_num_threads();
                const int worker = omp_get_thread_num();
                if (worker >= width ||
                    sched_getaffinity(0, sizeof(cpu_set_t), &masks[worker]) != 0)
                    failed = 1;
            }
            if (failed || actual != width)
                throw std::runtime_error("Observed team changed its requested geometry");
            // Pool workers created before the fixture narrowed its caller keep
            // their wider original mask; a bootstrap pin would be a singleton.
            for (int worker = 0; worker < width; ++worker)
            {
                cpu_set_t covered{};
                CPU_AND(&covered, &admitted, &masks[worker]);
                if (!CPU_EQUAL(&admitted, &covered))
                    throw std::runtime_error("Unbound OpenMP worker was pinned without native core admission");
            }
        }

        // A nested caller is rejected as well, and unwinding preserves its mask.
        bool rejected = false, exception_restored = false;
#pragma omp parallel num_threads(1)
        {
            // No exception may cross the OpenMP structured block, including a
            // failed fixture observation before or after production rejection.
            try
            {
                const auto before = callerMask();
                try { MPIBootstrapPhase::configureClusterRankThreads(OrchestrationConfig{}, NativeCPUPlacement::admission()); }
                catch (const std::runtime_error &error)
                {
                    rejected = std::string(error.what()).find("native CPU core admission") != std::string::npos;
                    const auto after = callerMask();
                    exception_restored = CPU_EQUAL(&before, &after);
                }
            }
            catch (...)
            {
                rejected = false;
                exception_restored = false;
            }
        }
        if (!rejected || !exception_restored)
            throw std::runtime_error("Nested bootstrap rejection lost exact caller ownership");
        std::cout << "PASS unbound bootstrap rejection: auto/full/partial/repeated/nested; caller, child and workers retain the rank mask" << std::endl;
        return 0;
    }

    /**
     * @brief CPU-bearing NUMA nodes of this host, via the same per-node CPU
     * observation the production allocator applies before admitting compute
     * placement. Memory-only nodes stay in the inventory as storage
     * endpoints; they cannot host a launchable compute rank.
     */
    std::vector<int> cpuBearingNodes()
    {
        std::vector<int> nodes;
        if (numa_available() < 0)
            return nodes;
        // The per-node CPU observation requires a bitmask sized for the CPU
        // count, not the node count: numa_node_to_cpus rejects smaller masks.
        auto *cpus = numa_bitmask_alloc(static_cast<unsigned int>(
            std::max(numa_num_possible_cpus(), 1)));
        if (!cpus)
            return nodes;
        for (int node = 0; node <= numa_max_node(); ++node)
        {
            if (numa_node_to_cpus(node, cpus) != 0)
                continue;
            if (numa_bitmask_weight(cpus) > 0)
                nodes.push_back(node);
        }
        numa_bitmask_free(cpus);
        return nodes;
    }

    inline std::atomic<int> diag_rank{0};
    /**
     * @brief Read-only OpenMP/OS place-state capture for collision attribution.
     *
     * Records the calling thread's OS affinity, libgomp logical place, that
     * place's full processor-id list (duplicates and ordering included), the
     * team configuration, and then runs one observation team with the same
     * shape the verifier uses, reporting each worker's own mask. No affinity
     * is changed and no team configuration is touched.
     */
    void capturePlaceState(const char *stage)
    {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        const bool have_mask = sched_getaffinity(0, sizeof(mask), &mask) == 0;
        const int place = omp_get_place_num();
        std::string place_ids;
        if (place >= 0)
        {
            const int size = omp_get_place_num_procs(place);
            std::vector<int> ids(static_cast<std::size_t>(size));
            omp_get_place_proc_ids(place, ids.data());
            for (const int id : ids)
                place_ids += " " + std::to_string(id);
        }
        std::fprintf(stderr,
            "[place-diag] stage=%s rank=%d tid=%ld mask_count=%d place=%d "
            "place_proc_count=%d place_proc_ids=[%s] max_threads=%d "
            "num_places=%d proc_bind=%d dynamic=%d\n",
            stage, diag_rank.load(), static_cast<long>(syscall(SYS_gettid)),
            have_mask ? CPU_COUNT(&mask) : -1, place,
            place >= 0 ? omp_get_place_num_procs(place) : -1,
            place_ids.c_str(), omp_get_max_threads(), omp_get_num_places(),
            omp_get_proc_bind(), omp_get_dynamic());
        // Full place-list enumeration: every place's complete processor-id
        // list in libgomp's own ordering, exposing any duplicate cpu across
        // places or empty/mispopulated entries.
        if (diag_rank.load() == 0)
        {
            const int places = omp_get_num_places();
            for (int p = 0; p < places; ++p)
            {
                const int count = omp_get_place_num_procs(p);
                std::vector<int> ids(static_cast<std::size_t>(count));
                omp_get_place_proc_ids(p, ids.data());
                std::string listed;
                for (const int id : ids)
                    listed += " " + std::to_string(id);
                std::fprintf(stderr, "[place-list] place=%d procs=%d[%s]\n",
                    p, count, listed.c_str());
            }
        }
        if (have_mask)
        {
            std::string listed;
            for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
                if (CPU_ISSET(cpu, &mask))
                    listed += " " + std::to_string(cpu);
            std::fprintf(stderr, "[place-diag] stage=%s mask=[%s]\n",
                stage, listed.c_str());
        }
        const int threads = omp_get_max_threads();
        std::vector<int> current(static_cast<std::size_t>(threads), -1);
        std::vector<cpu_set_t> masks(static_cast<std::size_t>(threads));
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            current[static_cast<std::size_t>(tid)] = sched_getcpu();
            if (sched_getaffinity(0, sizeof(cpu_set_t),
                    &masks[static_cast<std::size_t>(tid)]) != 0)
                CPU_ZERO(&masks[static_cast<std::size_t>(tid)]);
        }
        std::map<int, int> collision;
        for (int tid = 0; tid < threads; ++tid)
        {
            const int rep = current[static_cast<std::size_t>(tid)] >= 0
                ? MPIBootstrapPhase::physicalRepresentativeForCpu(
                    current[static_cast<std::size_t>(tid)])
                : -1;
            ++collision[rep];
            std::string listed;
            for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
                if (CPU_ISSET(cpu, &masks[static_cast<std::size_t>(tid)]))
                    listed += " " + std::to_string(cpu);
            std::fprintf(stderr,
                "[place-diag] stage=%s worker=%d current=%d mask_count=%d mask=[%s]\n",
                stage, tid, current[static_cast<std::size_t>(tid)],
                CPU_COUNT(&masks[static_cast<std::size_t>(tid)]), listed.c_str());
        }
        for (const auto &[rep, count] : collision)
            if (count > 1)
                std::fprintf(stderr,
                    "[place-diag] stage=%s COLLISION representative=%d threads=%d\n",
                    stage, rep, count);
    }

    /** @brief Own only the unique, small hostfile created by this test. */
    class Hostfile
    {
    public:
        /**
         * @brief Declare one MPI slot per launchable compute endpoint: a
         * CPU-bearing NUMA node. Memory-only nodes cannot host a rank.
         */
        explicit Hostfile(int slots)
        {
            char pattern[] = "/tmp/llaminar-inventory-hosts-XXXXXX";
            const int descriptor = mkstemp(pattern);
            if (descriptor < 0) throw std::runtime_error("Cannot create inventory hostfile");
            close(descriptor);
            path = pattern;
            std::ofstream output(path);
            output << "localhost slots=" << slots << '\n';
            if (!output) throw std::runtime_error("Cannot write inventory hostfile");
        }
        /** @brief Unlink the exact file created by this fixture. */
        ~Hostfile() { std::remove(path.c_str()); }
        std::string path;
    };

    /**
     * @brief Replay the actual retained service's first and repeated OpenMP teams.
     * @param cpu Original gathered rank CPU observation used by its existing PMA domain.
     * @param details Precise failed worker/partition invariant.
     * @return True only for the creator's whole physical-core budget and one
     *         unique admitted core place per worker; SMT siblings in a place are legal.
     */
    bool verifyRetainedServiceTeam(const PhysicalMemoryResource& cpu,std::string& details)
    try
    {
        cpu_set_t partition;
        CPU_ZERO(&partition);
        {
            const ScopedOpenMPPartitionAffinity admitted_partition;
            if(sched_getaffinity(0,sizeof(partition),&partition)!=0)
            {details="cannot observe admitted OpenMP partition";return false;}
        }
        std::set<int> physical;
        for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&partition))
            physical.insert(MPIBootstrapPhase::physicalRepresentativeForCpu(cpu));
        std::fprintf(stderr,"[cluster-diag] creator pid=%d omp_max_threads=%d "
            "partition_count=%d places=%d proc_bind=%d dynamic=%d\n",
            static_cast<int>(getpid()),omp_get_max_threads(),
            CPU_COUNT(&partition),omp_get_num_places(),omp_get_proc_bind(),
            omp_get_dynamic());
        const int expected=omp_get_max_threads();
        if(physical.empty()||physical.contains(-1)||expected!=static_cast<int>(physical.size()))
        {details="pre-main core place collapsed the rank's physical worker budget";return false;}
        const auto program=test::RetainedHostServiceReplayAccess::program(RetainedHostServiceRole::ExecutorParentTicket);
        auto memory=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(program.bom(cpu)),cpu.world_rank);
        auto owner=PreparedRetainedHostService::materialize(test::RetainedHostServiceReplayAccess::claim(memory,program));
        if(cpu.world_rank==0)capturePlaceState("after-retained-materialize");
        for(int repeat=0;repeat<2;++repeat)
        {
            struct Observation {int expected,maximum=0,dynamic=-1,actual=0;std::vector<cpu_set_t> masks;std::atomic<bool> failed{false};};
            Observation observed{expected,0,-1,0,std::vector<cpu_set_t>(static_cast<std::size_t>(expected))};
            const auto ticket=owner->worker().dispatch([](void* opaque){
                auto& value=*static_cast<Observation*>(opaque);
                cpu_set_t worker_mask;CPU_ZERO(&worker_mask);
                const bool worker_affinity_ok=sched_getaffinity(0,sizeof(worker_mask),&worker_mask)==0;
                std::fprintf(stderr,"[cluster-diag] retained_worker pid=%d tid=%ld "
                    "affinity_ok=%d mask_count=%d omp_max_threads=%d places=%d proc_bind=%d dynamic=%d\n",
                    static_cast<int>(getpid()),static_cast<long>(syscall(SYS_gettid)),
                    worker_affinity_ok?1:0,CPU_COUNT(&worker_mask),
                    omp_get_max_threads(),omp_get_num_places(),omp_get_proc_bind(),
                    omp_get_dynamic());
                value.maximum=omp_get_max_threads();value.dynamic=omp_get_dynamic();
#pragma omp parallel
                {
#pragma omp single
                    {
                        value.actual=omp_get_num_threads();
                        std::fprintf(stderr,"[cluster-diag] retained_team threads=%d "
                            "max=%d places=%d proc_bind=%d dynamic=%d\n",
                            value.actual,omp_get_max_threads(),omp_get_num_places(),
                            omp_get_proc_bind(),omp_get_dynamic());
                    }
                    const int worker=omp_get_thread_num();
                    if(worker>=value.expected || sched_getaffinity(0,sizeof(cpu_set_t),&value.masks[worker])!=0)
                        value.failed.store(true,std::memory_order_relaxed);
                    else
                        std::fprintf(stderr,"[cluster-diag] retained_team tid=%d worker=%d "
                            "current_cpu=%d place_num=%d mask_count=%d\n",
                            static_cast<int>(syscall(SYS_gettid)),worker,sched_getcpu(),
                            omp_get_place_num(),CPU_COUNT(&value.masks[worker]));
                }
                return !value.failed.load()&&value.maximum==value.expected&&value.actual==value.expected&&value.dynamic==0;
            },&observed);
            if(!owner->worker().await(ticket))
            {details="retained pthread changed the original program declaration's OpenMP budget";return false;}
            std::set<int> assigned;
            for(const auto& mask:observed.masks)
            {
                std::set<int> place;
                for(int id=0;id<CPU_SETSIZE;++id)if(CPU_ISSET(id,&mask))
                {
                    if(!CPU_ISSET(id,&partition))
                    {details="retained worker escaped its original rank partition";return false;}
                    place.insert(MPIBootstrapPhase::physicalRepresentativeForCpu(id));
                }
                if(place.size()!=1||place.contains(-1)||!assigned.insert(*place.begin()).second)
                {details="retained OpenMP workers do not own unique physical core places";return false;}
            }
        }
        details="retained creator budget and unique physical core places verified";
        return true;
    }
    catch(const std::exception& error)
    {
        details=error.what();
        return false;
    }

    /** @brief Enter production MPI bootstrap and prove each child's inventory. */
    int probe(int argc, char **argv)
    {
        // The private fixture flag is consumed at THIS parser boundary only:
        // the production parser never sees it, while the FULL original argv
        // (flag included) is forwarded to the self-launch so the RANK keeps
        // its intent (--placement-contract) and routes to the MPI rank path
        // instead of re-entering the launcher default path.
        std::vector<char *> parser_argv(argv, argv + argc);
        int parser_argc = argc;
        const bool fixture_flag = parser_argc > 1 &&
            std::string(parser_argv[1]) == "--placement-contract";
        if (fixture_flag)
        {
            parser_argv.erase(parser_argv.begin() + 1);
            --parser_argc;
        }
        auto config = OrchestrationConfigParser{}.parseArgs(parser_argc,
            parser_argv.data());
        // Only the independent placement case requests a whole-host singleton.
        // The ordinary inventory case must retain the real hostfile membership.
        if (fixture_flag)
        {
            config.device_for_this_rank = llaminar2::GlobalDeviceAddress::cpu(0);
            config.device_for_this_rank_numa_explicit = false;
            config.cpu_global_tp_all_local = true;
            config.mpi_procs = 1;
        }
        const auto bootstrap = MPIBootstrapPhase{}.execute(config, argc, argv);
        if (bootstrap.action == BootstrapResult::Action::EXIT)
            return bootstrap.exit_code;
        int provided = 0;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS)
            return 1;
        int passed = provided >= MPI_THREAD_MULTIPLE;
        {
            auto context = MPIContextFactory::global();
            const auto admitted_affinity = NUMATopology::detectLocalNUMANode();
            diag_rank.store(context->rank());
            if (context->rank() == 0) capturePlaceState("after-mpi-init");
            const auto inventory_owner = gatherClusterInventory(context, config.hostfile);
            if (context->rank() == 0) capturePlaceState("after-gather");
            const auto &inventory = *inventory_owner;
            passed = passed && inventory_owner == context->clusterInventory();
        if (!passed) std::fprintf(stderr, "[bisect] failed-at-or-before-inventory-ownership\n");
            const auto &local_observation = inventory.ranks.at(context->rank());
            // Compare independent pre-discovery affinity with the canonical
            // publication. One bound rank is not an unbound whole-host scope.
            const int expected_numa = admitted_affinity.detection_succeeded
                ? admitted_affinity.local_numa_node : -1;
            if(local_observation.cpu.numa_node!=expected_numa) {
                std::cerr<<"inventory_cpu_numa rank="<<context->rank()
                         <<" actual="<<local_observation.cpu.numa_node
                         <<" expected="<<expected_numa<<std::endl;
                passed=false;
            }
            // Exercise the serving specialization on the same observed rank
            // locality. MPI ordinal must not become an independent NUMA source.
            RuntimeInitPhase::resolveCPUShorthand(
                config, context->rank(), context->world_size(), admitted_affinity);
            passed = passed && config.device_map.size() == 1 &&
                config.device_map.front().first == context->rank() &&
                config.device_map.front().second.numa_node == admitted_affinity.local_numa_node;
            std::fprintf(stderr, "[bisect] device_map size=%zu first_numa=%d admitted_numa=%d\n",
                config.device_map.size(),
                config.device_map.empty() ? -1 : config.device_map.front().second.numa_node,
                admitted_affinity.local_numa_node);
        if (!passed) std::fprintf(stderr, "[bisect] failed-at-or-before-device_map\n");
            const auto topology = MPIBootstrap::detectCPUTopology();
            if (context->rank() == 0) capturePlaceState("after-plan-building");
            std::string detail;
            // READ-ONLY diagnosis for the startup-affinity attribution: no
            // affinity or team startup is altered by any of these observations.
            std::fprintf(stderr,
                "[cluster-diag] rank=%d pid=%d omp_max_threads=%d num_places=%d "
                "proc_bind=%d dynamic=%d\n",
                context->rank(), static_cast<int>(getpid()),
                omp_get_max_threads(), omp_get_num_places(),
                omp_get_proc_bind(), omp_get_dynamic());
            {
                std::fprintf(stderr, "[cluster-diag] omp_env:");
                for (const char *name : {"OMP_NUM_THREADS", "OMP_PLACES",
                                          "OMP_PROC_BIND", "OMP_DYNAMIC",
                                          "GOMP_SPINCOUNT"})
                {
                    const char *value = std::getenv(name);
                    std::fprintf(stderr, " %s=%s", name, value ? value : "<unset>");
                }
                std::fprintf(stderr, "\n");
                std::FILE *maps = std::fopen("/proc/self/maps", "r");
                char line[512];
                while (maps && std::fgets(line, sizeof(line), maps))
                    if (std::strstr(line, "libgomp") || std::strstr(line, "libiomp"))
                        std::fprintf(stderr, "[cluster-diag] omp_runtime %s", line);
                if (maps) std::fclose(maps);
                cpu_set_t process_mask;
                CPU_ZERO(&process_mask);
                if (sched_getaffinity(0, sizeof(process_mask), &process_mask) == 0)
                {
                    std::string listed;
                    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
                        if (CPU_ISSET(cpu, &process_mask))
                            listed += " " + std::to_string(cpu);
                    std::fprintf(stderr, "[cluster-diag] process_affinity_count=%d%s\n",
                        CPU_COUNT(&process_mask), listed.c_str());
                }
            }
            const bool affinity_ok = MPIBootstrapPhase::verifyStartupThreadAffinity(-1, true, detail);
            std::fprintf(stderr, "[cluster-diag] startup_affinity_ok=%d detail=%s\n",
                affinity_ok ? 1 : 0, detail.c_str());
            if (context->rank() == 0) capturePlaceState("after-startup-verify");
            {
                // Independent read-only team observation (own omp parallel,
                // same shape the verifier uses): per-thread current cpu and
                // full affinity mask, then the physical representatives.
                const int threads = omp_get_max_threads();
                std::vector<int> current(threads, -1);
                std::vector<cpu_set_t> masks(threads);
#pragma omp parallel
                {
                    const int tid = omp_get_thread_num();
                    current[tid] = sched_getcpu();
                    if (sched_getaffinity(0, sizeof(cpu_set_t), &masks[tid]) != 0)
                        CPU_ZERO(&masks[tid]);
                }
                std::map<int, int> representatives;
                for (int tid = 0; tid < threads; ++tid)
                {
                    std::string listed;
                    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
                        if (CPU_ISSET(cpu, &masks[tid]))
                            listed += " " + std::to_string(cpu);
                    const int rep = current[tid] >= 0
                        ? MPIBootstrapPhase::physicalRepresentativeForCpu(current[tid])
                        : -1;
                    ++representatives[rep];
                    std::fprintf(stderr,
                        "[cluster-diag] team tid=%d current_cpu=%d rep=%d mask_count=%d%s\n",
                        tid, current[tid], rep,
                        CPU_COUNT(&masks[tid]), listed.c_str());
                }
                for (const auto &[rep, count] : representatives)
                    std::fprintf(stderr, "[cluster-diag] representative=%d threads=%d\n",
                        rep, count);
            }
            std::string retained_detail;
            const PhysicalMemoryResource observed_cpu{context->rank(),DeviceId::cpu(),
                local_observation.cpu.memory_bytes,local_observation.cpu.free_memory_bytes};
            const bool retained_affinity_ok=verifyRetainedServiceTeam(observed_cpu,retained_detail);
            std::fprintf(stderr,"[cluster-diag] retained_affinity_ok=%d detail=%s\n",
                retained_affinity_ok?1:0,retained_detail.c_str());
            passed=passed&&retained_affinity_ok;
            if(!retained_affinity_ok)std::cerr<<"retained_team_affinity rank="<<context->rank()<<" "<<retained_detail<<std::endl;
            // The launched membership covers exactly the launchable compute
            // endpoints (CPU-bearing nodes); memory-only nodes remain in the
            // topology as storage inventory and never host a rank.
            const int expected_members = fixture_flag ? 1 : static_cast<int>(cpuBearingNodes().size());
            passed = passed && inventory.world_size == expected_members &&
                inventory.ranks.size() == static_cast<std::size_t>(context->world_size()) &&
                inventory.node_count == 1 && inventory.total_gpus == 0 && affinity_ok;
        if (!passed) std::fprintf(stderr, "[bisect] failed-at-or-before-inventory-world_size\n");
            for (int rank = 0; rank < context->world_size(); ++rank)
            {
                const auto &observation = inventory.ranks.at(rank);
                passed = passed && observation.rank == rank &&
                    observation.cpu_cores > 0 && observation.cpu.free_memory_bytes > 0;
            }
            if (context->world_size() > 1)
            {
                // One incorrect wrapper must fail collectively even when it
                // claims this is a local-only communicator. Later reads must
                // retain the first failure without re-entering discovery.
                MPIContext malformed(context->rank(), context->is_root() ? 1 : context->world_size(),
                                     context->communicator());
                for (int attempt = 0; attempt < 20; ++attempt)
                {
                    bool rejected = false;
                    try { malformed.clusterInventory(); }
                    catch (const std::runtime_error &error)
                    {
                        rejected = std::string(error.what()).find("context identity disagrees") != std::string::npos;
                    }
                    passed = passed && rejected;
                    MPI_Allreduce(MPI_IN_PLACE, &passed, 1, MPI_INT, MPI_MIN, context->communicator());
                }

                // A distinct, reversed communicator owns its own rank namespace.
                // Physical CPU affinity must survive that reversal unchanged.
                MPI_Comm reversed = MPI_COMM_NULL;
                if (MPI_Comm_split(context->communicator(), 0,
                                   context->world_size() - context->rank(), &reversed) != MPI_SUCCESS)
                    throw std::runtime_error("Cannot construct reversed inventory proof communicator");
                {
                    int reverse_rank = -1;
                    MPI_Comm_rank(reversed, &reverse_rank);
                    MPIContext reordered(reverse_rank, context->world_size(), reversed);
                    const auto observed = reordered.clusterInventory();
                    passed = passed && observed.get() != context->clusterInventory().get() &&
                        reverse_rank == context->world_size() - 1 - context->rank() &&
                        observed->ranks.at(reverse_rank).cpu.numa_node == local_observation.cpu.numa_node;
                    passed = passed && &reordered.concrete_topology().clusterInventory() == observed.get();

                    // Compile the actual production placement path, not merely
                    // the inventory projection. Rank order and NUMA order now
                    // differ, which exposes callers reconstructing locality.
                    DomainDefinition cpu_domain;
                    cpu_domain.name = "cpu_tp";
                    cpu_domain.scope = TPScope::NODE_LOCAL;
                    cpu_domain.backend = CollectiveBackendType::UPI;
                    for (const auto &peer : observed->ranks)
                    {
                        cpu_domain.devices.push_back(GlobalDeviceAddress::cpu(peer.cpu.numa_node));
                        cpu_domain.explicit_ranks.push_back(peer.rank);
                    }
                    OrchestrationConfig placement;
                    placement.domain_definitions = {cpu_domain};
                    const auto plans = ExecutionPlanBuilder{}.buildAllPlans(
                        placement, ModelConfig::qwen2_7b(), *observed);
                    for (const auto &plan : plans)
                        passed = passed && plan.numa_node == observed->ranks[plan.rank].cpu.numa_node &&
                            plan.primary_device.numa_node == observed->ranks[plan.rank].cpu.numa_node;

                    // The pipeline has its own stage ordering, independent of
                    // both physical socket order and communicator rank order.
                    // It must project the canonical inventory rather than
                    // treating different ranks as different physical hosts.
                    OrchestrationConfig pipeline;
                    const int entry_rank = observed->world_size - 1;
                    for (int stage = 0; stage < 2; ++stage)
                    {
                        const int owner = stage == 0 ? entry_rank : 0;
                        DomainDefinition domain;
                        domain.name = stage == 0 ? "entry" : "terminal";
                        domain.scope = TPScope::RANK_LOCAL;
                        domain.owner_rank = owner;
                        domain.devices = {GlobalDeviceAddress::cpu(observed->ranks[owner].cpu.numa_node)};
                        pipeline.domain_definitions.push_back(std::move(domain));
                    }
                    pipeline.pp_stage_definitions = {
                        PPStageDefinition::parse("0=entry:0-13"),
                        PPStageDefinition::parse("1=terminal:14-27")};
                    const auto pipeline_topology = ExecutionPlanBuilder{}.buildGlobalPPTopology(
                        pipeline, ModelConfig::qwen2_7b(), *observed);
                    passed = passed && pipeline_topology.transfers.size() == 1;
                    for (const auto &edge : pipeline_topology.transfers)
                    {
                        const auto &connection = edge.physicalConnection();
                        passed = passed && connection.sourceRank() == entry_rank &&
                            connection.destinationRank() == 0 &&
                            connection.locality() == RankConnectionLocality::SameNode &&
                            connection.sourceNode() == observed->ranks[entry_rank].node_id &&
                            connection.destinationNode() == observed->ranks[0].node_id;
                    }
                    const auto local_pipeline = GlobalPPRankPlanBuilder::build(pipeline_topology, reverse_rank);
                    for (const auto *action : local_pipeline.transferActions())
                    {
                        const auto &connection = action->physicalConnection(reverse_rank);
                        passed = passed && connection.sourceRank() == entry_rank &&
                            connection.destinationRank() == 0;
                    }

                    // Overlay must discover the same physical CPU owners from
                    // unpinned intent. It may not inherit the explicit map just
                    // validated above or substitute local-rank arithmetic.
                    auto routed = RoutedExpertDomain::fromExecutionDomainDefinition(
                        cpu_domain.toExecutionDomainDefinition());
                    routed.world_ranks.clear();
                    routed.owner_rank = -1;
                    routed.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
                    MoERoutedExpertPlacementPlan overlay;
                    overlay.enabled = true;
                    overlay.topology = RoutedExpertPlacementTopology::SingleDomain;
                    overlay.continuation_domain = overlay.shared_expert_domain = routed.name;
                    overlay.domains = {routed};
                    overlay.routed_tiers = {RoutedExpertTier{
                        .name = "priority_0", .domain = routed.name, .priority = 0, .fallback = true}};
                    const auto bound = bindMoEExpertOverlayPlanToClusterInventory(overlay, *observed);
                    passed = passed && bound->domains.front().world_ranks == cpu_domain.explicit_ranks;
                }
                MPI_Comm_free(&reversed);
            }
            // The topology must consume the same discovery authority, not
            // reconstruct zero-byte CPUs or environment-named GPUs. Only rank
            // zero reads the accessor here: that read must be collective-free.
            const auto publication = context->clusterInventory();
            const auto server_membership = serverRankMembershipTags(
                *publication, context->rank(), context->world_size() - 1);
            passed = passed && server_membership.at("rank") == std::to_string(context->rank()) &&
                server_membership.at("authority_rank") == std::to_string(context->world_size() - 1) &&
                server_membership.at("node_id") == std::to_string(local_observation.node_id) &&
                server_membership.at("local_rank") == std::to_string(local_observation.local_rank) &&
                server_membership.at("hostname") == local_observation.hostname;
            MPITopology mpi_topology(*context);
            passed = passed && &mpi_topology.clusterInventory() == publication.get();
            const auto *context_topology = context->topology();
            passed = passed && context_topology &&
                &context_topology->clusterInventory() == publication.get();
            if (context->is_root())
            {
                // Followers immediately enter the terminal Allreduce below.
                // Any accidental rediscovery here mismatches that collective;
                // this is stronger than comparing two snapshots after two gathers.
                for (int read = 0; read < 20; ++read)
                {
                    passed = passed && context->clusterInventory().get() == publication.get() &&
                        context->topology() == context_topology;
                    const auto repeat = gatherClusterInventory(context, config.hostfile);
                    passed = passed && repeat.get() == &inventory && repeat == publication;
                    passed = passed && serverRankMembershipTags(*context->clusterInventory(),
                        context->rank(), context->world_size() - 1) == server_membership;
                }
                const auto &snapshot = mpi_topology.clusterInventory();
                passed = passed && snapshot.node_count == inventory.node_count &&
                    snapshot.ranks.size() == inventory.ranks.size();
                for (int peer = 0; peer < context->world_size(); ++peer)
                {
                    const auto &observed = snapshot.ranks.at(peer);
                    passed = passed && observed.cpu.memory_bytes == inventory.ranks[peer].cpu.memory_bytes &&
                        observed.cpu.numa_node == inventory.ranks[peer].cpu.numa_node &&
                        observed.gpus.empty();
                }
            }
            // Every observer must agree before this test is green. A root-only
            // print is not evidence that all hostfile slots actually participated.
            MPI_Allreduce(MPI_IN_PLACE, &passed, 1, MPI_INT, MPI_MIN, context->communicator());
            std::cout << (passed ? "PASS" : "FAIL") << " rank=" << context->rank()
                      << " inventory_ranks=" << inventory.ranks.size()
                      << " nodes=" << inventory.node_count
                      << " gpus=" << inventory.total_gpus
                      << " expected_ranks=" << cpuBearingNodes().size()
                      << " workers=" << omp_get_max_threads()
                      << " affinity=" << detail << std::endl;
        }
        llaminar2::GPUDeviceProcessLifetime::retireActive();
        MPI_Finalize();
        return passed ? 0 : 1;
    }
}

/** @brief Launch through a real hostfile or execute one initialized child. */
int main(int argc, char **argv)
{
    // Ownership starts before this frontend creates MPI or runtime state.
    if (auto terminal = llaminar2::MPISessionOwner::enterProcess(argc, argv)) return *terminal;
    llaminar2::GPUDeviceProcessLifetime gpu_process_lifetime;

    try
    {
        if (argc == 2 && std::string(argv[1]) == "--caller-affinity")
            return checkBootstrapCallerAffinity();
        if (argc > 1)
        {
            const bool placement_contract = std::string(argv[1]) == "--placement-contract";
            // The full argv (flag included) reaches probe: the flag is
            // consumed at probe's parser boundary, and the self-launch
            // forwards it to the rank so the rank's intent is preserved.
            const int code = probe(argc, argv);
            if (placement_contract && code == 0)
                std::cout << "PLACEMENT-CONTRACT verified: persistent pool and the "
                    "initial thread hold unique capacity-ordered places" << std::endl;
            return code;
        }
        // Enumeration geometry coverage on the real host: this Station
        // exposes memory-only NUMA nodes above its CPU-bearing nodes, so the
        // CPU-bearing enumeration must be strictly smaller than the full node
        // set here, and every launchable slot must name a CPU-bearing node.
        // A host whose nodes all bear CPUs trivially satisfies both.
        const auto launchable = cpuBearingNodes();
        const int total_nodes = MPIBootstrap::detectCPUTopology().numa_nodes;
        const bool enumeration_ok = !launchable.empty() &&
            static_cast<int>(launchable.size()) <= total_nodes &&
            std::all_of(launchable.begin(), launchable.end(),
                        [](int node) { return node >= 0; });
        if (!enumeration_ok)
        {
            std::cerr << "CPU-bearing enumeration exceeds the node set or names a negative node" << std::endl;
            return 1;
        }
        const Hostfile file(static_cast<int>(launchable.size()));
        const auto child = fork();
        if (child < 0) throw std::runtime_error("Cannot fork inventory bootstrap");
        if (child == 0)
        {
            execl(argv[0], argv[0], "--hostfile", file.path.c_str(), "--device", "cpu", nullptr);
            _exit(127);
        }
        int status = 0;
        pid_t waited;
        do { waited = waitpid(child, &status, 0); }
        while (waited < 0 && errno == EINTR);
        return waited == child && WIFEXITED(status) ? WEXITSTATUS(status) : 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
