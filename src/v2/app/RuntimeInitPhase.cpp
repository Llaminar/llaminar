/**
 * @file RuntimeInitPhase.cpp
 * @brief Post-MPI runtime initialization
 *
 * Bootstrap observations specialize user intent before CPU/GPU allocator
 * singletons and captured runners are created. Physical CPU locality comes
 * from affinity, never from the numeric MPI rank or a hostfile slot order.
 * Model-aware placement belongs to the shared rank resolver. Startup reports
 * the runner's resolved configuration instead of binding an overlay itself.
 * Saved rank selection is interpreted before CPU placement. Automatic search
 * runs once on discovery root and publishes a complete apply document. Both
 * selections enter the same MPI admission transaction before runner creation.
 * Excluded discovery processes retire without acquiring an inference runner.
 */

#include "app/RuntimeInitPhase.h"
#include "app/MPIBootstrapPhase.h"
#include "app/ChatTemplateResolver.h"
#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "backends/InventoryPrinter.h"
#include "config/OrchestrationConfigParser.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include "execution/runner/RankInitializationLifecycle.h"
#include <type_traits>
#include "models/IGraphConfigBuilder.h"
#include "planning/ClusterInventoryGatherer.h"
#include "utils/ChatTemplate.h"
#include "utils/Tokenizer.h"
#include "utils/Logger.h"
#include "utils/DebugEnv.h"
#include "utils/NUMATopology.h"
#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <omp.h>
#include <sstream>
#include <stdexcept>

namespace llaminar2
{
    RuntimeInitPhase::RuntimeInitPhase()
        : RuntimeInitPhase(AutomaticPlanningStartup::preparation()) {}

    RuntimeInitPhase::RuntimeInitPhase(AutomaticPlanningStartup::Prepare prepare)
        : prepare_(std::move(prepare))
    {
        if (!prepare_) throw std::invalid_argument("Frontend initialization requires evidence preparation");
    }

    void RuntimeInitPhase::resolveCPUShorthand(
        OrchestrationConfig &config, int mpi_rank, int mpi_world_size,
        const NUMAInfo &numa_info)
    {
        if (!config.cpu_global_tp_all_local) return;
        if (mpi_world_size <= 0 || mpi_rank < 0 || mpi_rank >= mpi_world_size)
            throw std::runtime_error("CPU shorthand requires an admitted MPI rank");
        if (!numa_info.detection_succeeded || numa_info.local_numa_node < 0)
            throw std::runtime_error("CPU shorthand requires observed NUMA affinity");
        // Validate before mutating the configuration. The backend placement
        // resolver also checks any existing explicit identity on a repeat call.
        const int numa_node = resolveCPUBackendNUMANode(
            config, mpi_rank, mpi_world_size, numa_info);
        config.device_mode = DeviceAssignmentMode::EXPLICIT;
        config.device_map = {{mpi_rank, GlobalDeviceAddress::cpu(numa_node)}};
        config.device_map_numa_explicit = {{mpi_rank, true}};
        config.device_for_this_rank.reset();
        config.device_for_this_rank_numa_explicit = false;
        if (config.tp_scope == TPScope::AUTO) config.tp_scope = TPScope::GLOBAL;
        if (config.tp_degree <= 1) config.tp_degree = mpi_world_size;
    }

    int RuntimeInitPhase::resolveCPUBackendNUMANode(
        const OrchestrationConfig &config,
        int mpi_rank,
        int mpi_world_size,
        const NUMAInfo &numa_info)
    {
        std::optional<int> explicit_node;
        const auto declare_explicit_node =
            [&](int node, const char *source)
        {
            if (node < 0)
            {
                throw std::runtime_error(
                    std::string(source) +
                    " declared CPU NUMA placement without an exact node");
            }
            if (explicit_node.has_value() && *explicit_node != node)
            {
                throw std::runtime_error(
                    "Conflicting explicit CPU NUMA declarations for rank " +
                    std::to_string(mpi_rank));
            }
            explicit_node = node;
        };

        if (config.device_for_this_rank.has_value() &&
            config.device_for_this_rank->isCPU() &&
            config.device_for_this_rank_numa_explicit)
        {
            declare_explicit_node(
                config.device_for_this_rank->numa_node,
                "--device");
        }

        for (const auto &[mapped_rank, address] : config.device_map)
        {
            if (mapped_rank != mpi_rank || !address.isCPU())
                continue;

            const auto explicit_it = std::find_if(
                config.device_map_numa_explicit.begin(),
                config.device_map_numa_explicit.end(),
                [mapped_rank](const auto &entry)
                {
                    return entry.first == mapped_rank;
                });
            if (explicit_it != config.device_map_numa_explicit.end() &&
                explicit_it->second)
            {
                declare_explicit_node(address.numa_node, "--device-map");
            }
        }

        if (explicit_node.has_value())
        {
            if (numa_info.detection_succeeded &&
                numa_info.local_numa_node != *explicit_node)
            {
                throw std::runtime_error(
                    "Explicit CPU NUMA node " +
                    std::to_string(*explicit_node) +
                    " disagrees with rank affinity on node " +
                    std::to_string(numa_info.local_numa_node));
            }
            return *explicit_node;
        }

        if (mpi_world_size > 1 || config.cpu_global_tp_all_local)
        {
            if (!numa_info.detection_succeeded ||
                numa_info.local_numa_node < 0)
            {
                throw std::runtime_error(
                    "Rank-local CPU backend initialization requires exact NUMA affinity");
            }
            return numa_info.local_numa_node;
        }

        return -1;
    }

    bool RuntimeInitPhase::requiresHostWideAcceleratorVisibility(
        const OrchestrationConfig &config,
        int mpi_rank)
    {
        const auto intent = resolveOrchestrationIntent(config);
        if (const auto *automatic = std::get_if<AutomaticOrchestrationRequest>(&intent))
        {
            // A hostfile may provide just one rank on a multi-socket node.
            // Automatic search must see allowed accelerators on every socket,
            // not only the socket to which MPI happened to bind that rank.
            return automatic->allows(DeviceType::CUDA) || automatic->allows(DeviceType::ROCm);
        }
        const auto contains_gpu = [](const auto &devices)
        {
            return std::any_of(
                devices.begin(), devices.end(),
                [](const GlobalDeviceAddress &device)
                { return device.isGPU(); });
        };

        if (config.device_for_this_rank.has_value() &&
            config.device_for_this_rank->isGPU())
        {
            return true;
        }
        if (std::any_of(
                config.device_map.begin(), config.device_map.end(),
                [mpi_rank](const auto &entry)
                {
                    return entry.first == mpi_rank && entry.second.isGPU();
                }))
        {
            return true;
        }
        if (contains_gpu(config.tp_devices) || config.tp_degree > 1)
            return true;

        if (std::any_of(
                config.domain_definitions.begin(),
                config.domain_definitions.end(),
                [&](const DomainDefinition &domain)
                { return contains_gpu(domain.devices); }))
        {
            return true;
        }

        const auto &overlay = config.moe_routed_expert_plan;
        if (!overlay || !overlay->usesExpertOverlayAuthority())
            return false;

        if (std::any_of(
                overlay->domains.begin(), overlay->domains.end(),
                [&](const RoutedExpertDomain &domain)
                { return contains_gpu(domain.participants); }))
        {
            return true;
        }
        return std::any_of(
            overlay->dense_domains.begin(), overlay->dense_domains.end(),
            [&](const ExecutionDomainDefinition &domain)
            { return contains_gpu(domain.participants); });
    }

    void RuntimeInitPhase::initializeDiscoveryRuntime(
        const OrchestrationConfig &config,
        int discovery_rank,
        int discovery_world_size)
    {
        if (discovery_rank < 0 || discovery_world_size <= 0 ||
            discovery_rank >= discovery_world_size)
        {
            throw std::invalid_argument(
                "Discovery runtime initialization requires a valid communicator rank");
        }

        // The command and serving frontends must expose the same fixed worker
        // geometry to planning.  This changes neither affinity nor physical
        // inventory; it only installs an explicit user-requested team before
        // the inventory records it.
        MPIBootstrapPhase::configureRequestedCPUThreads(config);
        if (std::getenv("OMP_NUM_THREADS") == nullptr)
        {
            LOG_WARN("MPI runtime has no OMP_NUM_THREADS; use the canonical bootstrap for worker placement");
        }

        const auto numa = NUMATopology::detectLocalNUMANode();
        const auto execution_rank = config.execution_rank_selection
            ? config.execution_rank_selection->executionRank(discovery_rank)
            : std::optional<int>{discovery_rank};
        const int execution_size = config.execution_rank_selection
            ? config.execution_rank_selection->size()
            : discovery_world_size;
        const bool explicit_cpu = execution_rank && config.device_for_this_rank &&
            config.device_for_this_rank->isCPU() && config.device_for_this_rank_numa_explicit;
        if (execution_rank && (explicit_cpu || config.cpu_global_tp_all_local))
        {
            const int required = explicit_cpu ? config.device_for_this_rank->numa_node
                : (numa.detection_succeeded ? numa.local_numa_node : -1);
            std::string detail;
            if (!MPIBootstrapPhase::verifyStartupThreadAffinity(required, true, detail))
            {
                if (explicit_cpu || debugEnv().runtime_debug.assert_thread_affinity)
                {
                    throw std::runtime_error(
                        "Startup thread affinity verification failed: " + detail);
                }
                LOG_WARN("Startup thread affinity verification warning: " << detail);
            }
        }

        // CPU backend identity is an admission prerequisite even when this
        // command stops after planning.  Projection sampling allocates the
        // exact workspace ABI through IBackend; returning a host allocation
        // without this rank-local backend would measure an endpoint that
        // production inference cannot own.
        if (execution_rank)
        {
            initCPUBackend(resolveCPUBackendNUMANode(
                config, *execution_rank, execution_size, numa));
        }
        else
        {
            // Excluded saved-plan ranks still join discovery and must publish
            // their actual host observation.  They do not borrow another
            // execution rank's requested NUMA identity.
            initCPUBackend(numa.detection_succeeded ? numa.local_numa_node : -1);
        }

        // GPU visibility is an inventory fact, not a CPU-affinity heuristic.
        // A permitted accelerator backend therefore requires the full host
        // view before the planner can choose an owning rank or tier.
        const bool host_wide = std::getenv("LLAMINAR_SELF_BOOTSTRAPPED") != nullptr ||
            requiresHostWideAcceleratorVisibility(config, execution_rank.value_or(-1));
        DeviceManager::instance().initialize(
            host_wide ? -1 : numa.local_numa_node, false);
    }

    bool RuntimeInitPhase::runDryRunPreflight(IOrchestrationRunner &runner,
                                              int mpi_rank,
                                              std::ostream &out)
    {
        if (!runner.initializeForDryRun())
        {
            if (mpi_rank == 0) LOG_ERROR("--dry-run preflight failed: " << runner.lastError());
            return false;
        }
        if (mpi_rank == 0)
            out << "\n=== Resolved Orchestration Plan ===\n"
                << runner.config().toString() << std::endl;
        return true;
    }

    RuntimeInitResult RuntimeInitPhase::execute(OrchestrationConfig &config,
                                                int &argc, char **&argv)
    {
        const auto mpi_env = MPIBootstrap::detectMPIEnvironment();
        if (mpi_env.is_mpi_process &&
            debugEnv().mpi_bootstrap.ompi_mca_btl_vader_single_copy_mechanism.empty())
            setenv("OMPI_MCA_btl_vader_single_copy_mechanism", "none", 0);

        // Declared before every dependent owner. Early returns and exceptions
        // release those owners before this scope finalizes the process.
        auto session = MPIProcessSession::initialize(argc, argv);
        try
        {
            auto mpi_ctx = MPIContextFactory::global();
            Logger::getInstance().setRank(mpi_ctx->rank());
            std::uint32_t ordinal = 0;
            const auto phase = [&](std::string_view name, const auto &work)
            {
                const auto result = RankInitializationLifecycle::execute(
                    {ordinal++, name}, work, [&](auto identity, auto outcome)
                    {
                        return MPIRankInitializationConsensus::reach(
                            mpi_ctx->communicator(), identity, outcome);
                    });
                if (!result.succeeded())
                    throw std::runtime_error(result.diagnostic("Frontend initialization"));
            };

            phase("prepare_local_runtime", [&]
            {
                // MPI may modify argv. Preserve command intent across parsing;
                // malformed local input must reach consensus before discovery.
                const bool benchmark = config.benchmark_mode;
                const bool serve = config.serve_mode;
                config = OrchestrationConfigParser{}.parseArgs(argc, argv);
                config.benchmark_mode = config.benchmark_mode || benchmark;
                config.serve_mode = config.serve_mode || serve;
                if (config.model_path.empty())
                    throw std::invalid_argument("Model path required (-m)");

                const auto numa = NUMATopology::detectLocalNUMANode();
                const auto selected_rank = config.execution_rank_selection ?
                    config.execution_rank_selection->executionRank(mpi_ctx->rank()) :
                    std::optional<int>{mpi_ctx->rank()};
                const int selected_size = config.execution_rank_selection ?
                    config.execution_rank_selection->size() : mpi_ctx->world_size();

                // Materialize this legacy shorthand before the shared
                // discovery initializer observes CPU ownership.  The plan
                // command does not mutate its automatic request this way.
                if (selected_rank)
                {
                    resolveCPUShorthand(config, *selected_rank, selected_size, numa);
                }
                initializeDiscoveryRuntime(config, mpi_ctx->rank(), mpi_ctx->world_size());
                return true;
            });

            // Discovery already owns its collective failure/publication
            // transaction. Only observers consume this immutable inventory.
            const auto cluster = gatherClusterInventory(mpi_ctx, config.hostfile);

            if (std::holds_alternative<AutomaticOrchestrationRequest>(resolveOrchestrationIntent(config)))
            {
                // Evidence collection uses every discovery participant before
                // selection narrows membership. Plan and serve share this exact
                // source/measurement/selection/publication lifecycle.
                auto startup = AutomaticPlanningStartup::run(config, *cluster, mpi_ctx, prepare_);
                config = std::move(startup.applied);
                if (startup.root_selection)
                {
                    const auto &selected = *startup.root_selection;
                    LOG_INFO("[AutomaticPlanning] selected strategy="
                        << orchestrationStrategyName(selected.candidate().strategy())
                        << " proposed=" << selected.counts().proposed
                        << " evaluated=" << selected.counts().evaluated
                        << " request_seconds=" << selected.cost().requestSeconds());
                }
            }

            std::optional<ExecutionRankSelection> selection;
            phase("prepare_execution_selection", [&]
            {
                if (config.execution_rank_selection)
                {
                    if (!std::holds_alternative<ApplyOrchestrationRequest>(resolveOrchestrationIntent(config)))
                        throw std::invalid_argument("Saved execution ranks require apply intent");
                    selection = *config.execution_rank_selection;
                }
                else selection = ExecutionRankSelection::all(mpi_ctx->world_size());
                if (mpi_ctx->is_root()) InventoryPrinter::printClusterInventory(*cluster);
                return true;
            });
            // Every discovery rank enters the same existing admission protocol,
            // including unselected ranks and callers with identity membership.
            // It rejects asymmetric intent before splitting. The active context
            // retains discovery and its inventory; no second exchange occurs.
            auto admitted = MPIContextFactory::selectRanks(mpi_ctx, selection->discoveryRanks());
            if (std::holds_alternative<InactiveMPIRank>(admitted))
                return RuntimeInitExit::Completed;
            mpi_ctx = std::get<std::shared_ptr<MPIContext>>(std::move(admitted));
            Logger::getInstance().setRank(mpi_ctx->rank());

            std::unique_ptr<IOrchestrationRunner> runner;
            phase("construct_runner", [&]
            {
                auto factory = createOrchestrationRunnerFactory(mpi_ctx);
                runner = factory->createFromOrchestrationConfig(config);
                return runner != nullptr;
            });
            if (config.dry_run)
                return runDryRunPreflight(*runner, mpi_ctx->rank(), std::cout)
                    ? RuntimeInitExit::Completed : RuntimeInitExit::Failed;

            // Runner initialization owns graph/model phase consensus. This
            // frontend must not wrap it in a competing initialization protocol.
            if (!runner->initialize())
            {
                if (mpi_ctx->is_root()) LOG_ERROR("Failed to initialize: " << runner->lastError());
                return RuntimeInitExit::Failed;
            }

            OrchestrationConfig ready_config;
            std::shared_ptr<ITokenizer> tokenizer;
            phase("prepare_frontend_context", [&]
            {
                config = runner->config();
                ready_config = config;
                if (config.explain_placement && mpi_ctx->is_root())
                    std::cout << "\n=== Placement Explanation ===\n" << config.toString() << std::endl;
                tokenizer = runner->tokenizer();
                if (!tokenizer) throw std::runtime_error("Runner did not publish a tokenizer");
                ChatTemplateResolver::resolve(config.chat_template_override, tokenizer, mpi_ctx->rank());
                if (config.chat_template_override.empty() && !runner->architecture().empty())
                {
                    auto builder = createGraphConfigBuilder(runner->architecture());
                    if (builder)
                    {
                        const auto model_template = builder->chatTemplateOverride();
                        if (model_template && !model_template->empty())
                            tokenizer->setChatTemplate(ChatTemplate::create(*model_template, "", ""));
                    }
                }
                return true;
            });

            // All throwing preparation is complete before transferring the
            // finalizer. A failed partial aggregate must never finalize MPI
            // before a still-local runner; these moves are statically no-throw.
            static_assert(std::is_nothrow_move_constructible_v<OrchestrationConfig>);
            static_assert(std::is_nothrow_move_constructible_v<AppContext>);
            return AppContext{
                .mpi_session = std::move(session),
                .config = std::move(ready_config),
                .mpi_ctx = std::move(mpi_ctx),
                .runner = std::move(runner),
                .tokenizer = std::move(tokenizer)};
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(error.what());
            return RuntimeInitExit::Failed;
        }
    }

} // namespace llaminar2
