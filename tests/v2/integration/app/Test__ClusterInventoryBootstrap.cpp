/**
 * @file Test__ClusterInventoryBootstrap.cpp
 * @brief Model-free production hostfile bootstrap and inventory exchange proof.
 *
 * The parent creates a local hostfile and calls the real parser/bootstrap.
 * MPI must use every declared slot without inheriting a launcher-local CPU
 * set. Children prove their full physical worker binding, then collectively
 * gather the canonical CPU inventory. Repeated rank-zero reads and topology
 * construction must share that exact publication without another exchange.
 * Server evidence must project those same physical groups without a new probe
 * or collective, including when only the request authority reads the snapshot.
 * Production CPU domain/overlay binding must also preserve physical ownership
 * when communicator rank order is reversed. Pipeline activation edges and
 * endpoint actions retain that same immutable membership projection.
 * No model or accelerator is opened.
 * This proves a real multi-process node, not a remote-network deployment.
 * A separate unbound-runtime lifecycle check calls actual rank-worker admission without
 * MPI or devices, proving exact caller/child masks and retained worker binding
 * across automatic, partial and repeated teams and nested-call rejection.
 *
 * The custom entry owns one lexical native GPU lifetime without probing a
 * device. Native workers/caches retire before process exit and MPI finalization.
 */
#include "backends/GPUDeviceProcessLifetime.h"
#include "fixtures/execution/RetainedHostServiceReplayAccess.h"
#include "execution/local_execution/graph/RetainedParentTicketServiceWorker.h"
#include "utils/ScopedOpenMPPartitionAffinity.h"
#include "utils/ScopedCallerThreadAffinity.h"
#include "utils/MPISessionOwner.h"
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
     * @brief Check production rank admission with an initially unchanged partition mask.
     * @return Zero only for restored caller/child masks and persistent worker placement.
     * @throws std::runtime_error For missing native geometry or any failed invariant.
     *
     * Two distinct allowed physical cores expose the former master's singleton
     * leak without depending on CPU IDs, NUMA labels, MPI ranks or accelerators.
     * Only this test caller is narrowed; its original mask is restored on exit.
     */
    int checkBootstrapCallerAffinity()
    {
        const ScopedCallerThreadAffinity restore_test_caller;
        if (omp_get_proc_bind() != omp_proc_bind_false)
            throw std::runtime_error("Bootstrap caller lifecycle requires pre-exec OMP_PROC_BIND=false");
        const auto original = callerMask();
        std::map<int, int> physical;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &original))
                physical.try_emplace(MPIBootstrapPhase::physicalRepresentativeForCpu(cpu), cpu);
        if (physical.size() < 2 || physical.contains(-1))
            throw std::runtime_error("Bootstrap caller lifecycle requires two admitted physical cores");
        std::vector<int> cpus;
        cpu_set_t admitted{};
        for (const auto &[representative, cpu] : physical)
        {
            cpus.push_back(cpu);
            CPU_SET(cpu, &admitted);
            if (cpus.size() == 2) break;
        }
        if (sched_setaffinity(0, sizeof(admitted), &admitted) != 0)
            throw std::runtime_error("Cannot admit bootstrap lifecycle caller cores");
        {
            const ScopedOpenMPPartitionAffinity partition;
            const auto unchanged = callerMask();
            if (!CPU_EQUAL(&admitted, &unchanged))
                throw std::runtime_error("Unbound partition unexpectedly changed caller mask");
        }

        // Zero is production auto sizing. The one-worker request exercises a
        // partial team; full/default admissions then reuse its original pool.
        for (const int requested : {0, 2, 1, 2, 0})
        {
            OrchestrationConfig config;
            config.n_threads = requested;
            MPIBootstrapPhase::configureClusterRankThreads(config);
            const auto restored = callerMask();
            if (!CPU_EQUAL(&admitted, &restored))
                throw std::runtime_error("Production bootstrap leaked its master's singleton affinity");
            bool child_inherited = false;
            std::thread child([&] {
                cpu_set_t inherited{};
                child_inherited = sched_getaffinity(0, sizeof(inherited), &inherited) == 0 &&
                    CPU_EQUAL(&admitted, &inherited);
            });
            child.join();
            if (!child_inherited)
                throw std::runtime_error("Native child lost the restored original rank mask");
            const int expected = requested == 0 ? 2 : requested;
            if (omp_get_max_threads() != expected || omp_get_dynamic() != 0 ||
                omp_get_proc_bind() != omp_proc_bind_false)
                throw std::runtime_error("Bootstrap changed the admitted runtime worker policy");
            std::vector<cpu_set_t> masks(static_cast<std::size_t>(expected));
            int actual = 0, failed = 0;
#pragma omp parallel reduction(| : failed)
            {
#pragma omp single
                actual = omp_get_num_threads();
                const int worker = omp_get_thread_num();
                if (worker >= expected ||
                    sched_getaffinity(0, sizeof(cpu_set_t), &masks[worker]) != 0)
                    failed = 1;
            }
            if (failed || actual != expected || !CPU_EQUAL(&admitted, &masks[0]))
                throw std::runtime_error("Observed team changed its caller or worker geometry");
            for (int worker = 1; worker < expected; ++worker)
                if (CPU_COUNT(&masks[worker]) != 1 || !CPU_ISSET(cpus[worker], &masks[worker]))
                    throw std::runtime_error("Retained OpenMP worker lost its original singleton binding");
        }

        // The partition authority rejects nested callers before team binding.
        // Actual exception unwinding must also preserve that caller's mask.
        bool rejected = false, exception_restored = false;
#pragma omp parallel num_threads(1)
        {
            // No exception may cross the OpenMP structured block, including a
            // failed fixture observation before or after production rejection.
            try
            {
                const auto before = callerMask();
                try { MPIBootstrapPhase::configureClusterRankThreads(OrchestrationConfig{}); }
                catch (const std::runtime_error &error)
                {
                    rejected = std::string(error.what()).find("outer host thread") != std::string::npos;
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
        std::cout << "PASS bootstrap caller restoration: auto/full/partial/repeated/exception; retained workers; native child" << std::endl;
        return 0;
    }

    /** @brief Own only the unique, small hostfile created by this test. */
    class Hostfile
    {
    public:
        /** @brief Declare one MPI slot per discovered local NUMA endpoint. */
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
        const int expected=omp_get_max_threads();
        if(physical.empty()||physical.contains(-1)||expected!=static_cast<int>(physical.size()))
        {details="pre-main core place collapsed the rank's physical worker budget";return false;}
        const auto program=test::RetainedHostServiceReplayAccess::program(RetainedHostServiceRole::ExecutorParentTicket);
        auto memory=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(program.bom(cpu)),cpu.world_rank);
        auto owner=PreparedRetainedHostService::materialize(test::RetainedHostServiceReplayAccess::claim(memory,program));
        for(int repeat=0;repeat<2;++repeat)
        {
            struct Observation {int expected,maximum=0,dynamic=-1,actual=0;std::vector<cpu_set_t> masks;std::atomic<bool> failed{false};};
            Observation observed{expected,0,-1,0,std::vector<cpu_set_t>(static_cast<std::size_t>(expected))};
            const auto ticket=owner->worker().dispatch([](void* opaque){
                auto& value=*static_cast<Observation*>(opaque);
                value.maximum=omp_get_max_threads();value.dynamic=omp_get_dynamic();
#pragma omp parallel
                {
#pragma omp single
                    value.actual=omp_get_num_threads();
                    const int worker=omp_get_thread_num();
                    if(worker>=value.expected || sched_getaffinity(0,sizeof(cpu_set_t),&value.masks[worker])!=0)
                        value.failed.store(true,std::memory_order_relaxed);
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
        auto config = OrchestrationConfigParser{}.parseArgs(argc, argv);
        const auto bootstrap = MPIBootstrapPhase{}.execute(config, argc, argv);
        if (bootstrap.action == BootstrapResult::Action::EXIT)
            return bootstrap.exit_code;
        int provided = 0;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS)
            return 1;
        int passed = provided >= MPI_THREAD_MULTIPLE;
        {
            auto context = MPIContextFactory::global();
            const auto inventory_owner = gatherClusterInventory(context, config.hostfile);
            const auto &inventory = *inventory_owner;
            passed = passed && inventory_owner == context->clusterInventory();
            const auto &local_observation = inventory.ranks.at(context->rank());
            // A lone rank deliberately owns the whole host in the inventory
            // (numa_node -1, e.g. a single-NUMA GB10). Serving then resolves
            // the shorthand from the process's observed affinity, exactly as
            // RuntimeInitPhase does; distributed ranks use the inventory.
            const bool whole_host = context->world_size() == 1;
            passed = passed && (!whole_host || local_observation.cpu.numa_node == -1);
            const NUMAInfo admitted_affinity = whole_host ? NUMATopology::detectLocalNUMANode() : NUMAInfo{
                .local_numa_node = local_observation.cpu.numa_node,
                .total_numa_nodes = local_observation.numa_nodes,
                .detection_succeeded = local_observation.cpu.numa_node >= 0,
                .detection_method = "canonical-inventory"};
            // Exercise the serving specialization on the same observed rank
            // locality. MPI ordinal must not become an independent NUMA source.
            RuntimeInitPhase::resolveCPUShorthand(
                config, context->rank(), context->world_size(), admitted_affinity);
            passed = passed && config.device_map.size() == 1 &&
                config.device_map.front().first == context->rank() &&
                config.device_map.front().second.numa_node == admitted_affinity.local_numa_node;
            const auto topology = MPIBootstrap::detectCPUTopology();
            std::string detail;
            const bool affinity_ok = MPIBootstrapPhase::verifyStartupThreadAffinity(-1, true, detail);
            std::string retained_detail;
            const PhysicalMemoryResource observed_cpu{context->rank(),DeviceId::cpu(),
                local_observation.cpu.memory_bytes,local_observation.cpu.free_memory_bytes};
            const bool retained_affinity_ok=verifyRetainedServiceTeam(observed_cpu,retained_detail);
            passed=passed&&retained_affinity_ok;
            if(!retained_affinity_ok)std::cerr<<"retained_team_affinity rank="<<context->rank()<<" "<<retained_detail<<std::endl;
            passed = passed && inventory.world_size == topology.numa_nodes &&
                inventory.ranks.size() == static_cast<std::size_t>(context->world_size()) &&
                inventory.node_count == 1 && inventory.total_gpus == 0 && affinity_ok;
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
                      << " expected_ranks=" << topology.numa_nodes
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
        if (argc > 1) return probe(argc, argv);
        const Hostfile file(MPIBootstrap::detectCPUTopology().numa_nodes);
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
