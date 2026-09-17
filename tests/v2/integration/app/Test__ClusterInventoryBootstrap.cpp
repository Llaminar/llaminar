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
 */
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
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace llaminar2;

namespace
{
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
            const NUMAInfo admitted_affinity{
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
                config.device_map.front().second.numa_node == local_observation.cpu.numa_node;
            const auto topology = MPIBootstrap::detectCPUTopology();
            std::string detail;
            const bool affinity_ok = MPIBootstrapPhase::verifyStartupThreadAffinity(-1, true, detail);
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
        MPI_Finalize();
        return passed ? 0 : 1;
    }
}

/** @brief Launch through a real hostfile or execute one initialized child. */
int main(int argc, char **argv)
{
    try
    {
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
