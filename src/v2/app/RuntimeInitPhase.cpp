/**
 * @file RuntimeInitPhase.cpp
 * @brief Post-MPI runtime initialization
 */

#include "app/RuntimeInitPhase.h"
#include "app/MPIBootstrapPhase.h"
#include "app/MPIShutdown.h"
#include "app/ChatTemplateResolver.h"
#include "backends/ComputeBackend.h"
#include "backends/InventoryPrinter.h"
#include "config/OrchestrationConfigParser.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include "models/IGraphConfigBuilder.h"
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

namespace llaminar2
{

    bool RuntimeInitPhase::runDryRunPreflight(OrchestrationConfig &config,
                                              IOrchestrationRunner &runner,
                                              int mpi_rank,
                                              std::ostream &out)
    {
        if (!runner.initializeForDryRun())
        {
            if (mpi_rank == 0)
            {
                LOG_ERROR("--dry-run preflight failed: " << runner.lastError());
            }
            config.dry_run = false;
            runner.shutdown();
            return false;
        }

        if (mpi_rank == 0)
        {
            out << "\n=== Resolved Orchestration Plan ===\n"
                << runner.config().toString() << std::endl;
        }

        runner.shutdown();
        return true;
    }

    std::optional<AppContext> RuntimeInitPhase::execute(OrchestrationConfig &config,
                                                        int &argc, char **&argv)
    {
        // OpenMPI vader CMA workaround
        MPIEnvironmentInfo mpi_env = MPIBootstrap::detectMPIEnvironment();
        if (mpi_env.is_mpi_process && debugEnv().mpi_bootstrap.ompi_mca_btl_vader_single_copy_mechanism.empty())
        {
            setenv("OMPI_MCA_btl_vader_single_copy_mechanism", "none", 0);
        }

        // Initialize MPI
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

        // Re-parse arguments (MPI_Init may modify argc/argv)
        OrchestrationConfigParser parser;
        config = parser.parseArgs(argc, argv);

        auto mpi_ctx = MPIContextFactory::global();

        // Honor --threads override (if > 0). MPIBootstrap has already set
        // OMP_NUM_THREADS based on topology; --threads lets the user force a
        // specific count. OpenMP-threaded BLAS backends (OpenBLAS, MKL) also
        // honor OMP_NUM_THREADS, so this single override propagates.
        if (config.n_threads > 0)
        {
            setenv("OMP_NUM_THREADS", std::to_string(config.n_threads).c_str(), 1);
            omp_set_num_threads(config.n_threads);
        }

        // Check OMP_NUM_THREADS
        if (std::getenv("OMP_NUM_THREADS") == nullptr)
        {
            LOG_WARN("[Main] Running under MPI without OMP_NUM_THREADS set. For strict per-rank core pinning, "
                     "launch with canonical wrapper or set OMP_NUM_THREADS/OMP_PLACES/OMP_PROC_BIND in the launcher.");
        }

        // Detect NUMA node for MPI rank
        auto numa_info = NUMATopology::detectLocalNUMANode();

        const bool cpu_explicit_numa_mode =
            config.device_for_this_rank.has_value() &&
            config.device_for_this_rank->isCPU() &&
            config.device_for_this_rank_numa_explicit;

        const bool cpu_global_mode = config.cpu_global_tp_all_local;

        // CPU affinity verification
        if (cpu_explicit_numa_mode || cpu_global_mode)
        {
            int required_numa = -1;
            if (cpu_explicit_numa_mode)
            {
                required_numa = config.device_for_this_rank->numa_node;
            }
            else if (numa_info.detection_succeeded)
            {
                required_numa = numa_info.local_numa_node;
            }

            std::string affinity_details;
            const bool affinity_ok = MPIBootstrapPhase::verifyStartupThreadAffinity(
                required_numa, true, affinity_details);

            const bool strict_assert = []()
            {
                return debugEnv().runtime_debug.assert_thread_affinity;
            }();

            if (!affinity_ok)
            {
                const bool fail_fast = strict_assert || cpu_explicit_numa_mode;
                if (fail_fast)
                {
                    LOG_ERROR("[Main] Startup thread affinity verification failed: " << affinity_details);
                    LOG_ERROR("[Main] Fix launcher pinning (mpirun binding/cpu-set) or set LLAMINAR_ASSERT_THREAD_AFFINITY=0 to downgrade to warning");
                    mpiShutdown();
                    return std::nullopt;
                }

                LOG_WARN("[Main] Startup thread affinity verification warning: " << affinity_details);
            }
            else
            {
                LOG_DEBUG("[Main] Startup thread affinity verification passed");
            }
        }

        // CPU shorthand runtime semantics
        if (config.cpu_global_tp_all_local)
        {
            int local_rank = 0;
            MPI_Comm local_comm = MPI_COMM_NULL;
            if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_ctx->rank(), MPI_INFO_NULL, &local_comm) == MPI_SUCCESS)
            {
                MPI_Comm_rank(local_comm, &local_rank);
                MPI_Comm_free(&local_comm);
            }

            config.device_mode = DeviceAssignmentMode::EXPLICIT;
            config.device_map.clear();
            config.device_map.emplace_back(mpi_ctx->rank(), GlobalDeviceAddress::cpu(local_rank));
            config.device_map_numa_explicit.clear();
            config.device_map_numa_explicit.emplace_back(mpi_ctx->rank(), true);

            // Clear the original -d flag to avoid validation conflicts
            // (device_map now owns the per-rank assignment)
            config.device_for_this_rank = std::nullopt;
            config.device_for_this_rank_numa_explicit = false;

            if (config.tp_scope == TPScope::AUTO)
            {
                config.tp_scope = TPScope::GLOBAL;
            }
            if (config.tp_degree <= 1)
            {
                config.tp_degree = mpi_ctx->world_size();
            }

            if (mpi_ctx->rank() == 0)
            {
                LOG_DEBUG("[Main] CPU shorthand runtime mapping enabled: GLOBAL TP degree="
                          << config.tp_degree << ", world_size=" << mpi_ctx->world_size());
            }
        }

        // Set logger rank
        Logger::getInstance().setRank(mpi_ctx->rank());

        if (mpi_ctx->rank() == 0)
        {
            LOG_DEBUG("=== NUMA Topology Detection ===");
        }

        LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] NUMA node: " << numa_info.local_numa_node
                           << " (detection: " << numa_info.detection_method << ")");

        if (!numa_info.detection_succeeded && mpi_ctx->rank() == 0)
        {
            LOG_WARN("NUMA detection failed, using fallback node 0. This may impact multi-socket performance.");
        }

        // Initialize DeviceManager
        const bool self_bootstrapped = (std::getenv("LLAMINAR_SELF_BOOTSTRAPPED") != nullptr);

        int device_manager_numa_filter;
        if (self_bootstrapped)
        {
            device_manager_numa_filter = -1;
            LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Self-bootstrapped: initialising DeviceManager without NUMA filtering");
        }
        else
        {
            device_manager_numa_filter = numa_info.local_numa_node;

            std::optional<GlobalDeviceAddress> mapped_device_for_rank;
            for (const auto &[mapped_rank, mapped_addr] : config.device_map)
            {
                if (mapped_rank == mpi_ctx->rank())
                {
                    mapped_device_for_rank = mapped_addr;
                    break;
                }
            }

            const bool has_gpu_request =
                (config.device_for_this_rank.has_value() && config.device_for_this_rank->isGPU()) ||
                (mapped_device_for_rank.has_value() && mapped_device_for_rank->isGPU()) ||
                std::any_of(config.tp_devices.begin(), config.tp_devices.end(),
                            [](const GlobalDeviceAddress &d)
                            { return d.isGPU(); }) ||
                std::any_of(config.domain_definitions.begin(), config.domain_definitions.end(),
                            [](const DomainDefinition &dom)
                            {
                                return std::any_of(dom.devices.begin(), dom.devices.end(),
                                                   [](const GlobalDeviceAddress &d)
                                                   { return d.isGPU(); });
                            }) ||
                (config.tp_degree > 1);

            if (has_gpu_request)
            {
                device_manager_numa_filter = -1;
                LOG_INFO("[Main] GPU device(s) requested; initialising DeviceManager without NUMA filtering");
            }
        }

        auto &dm = DeviceManager::instance();
        dm.initialize(device_manager_numa_filter, false); // No local table printing

        // Build cluster inventory on ALL ranks (collective MPI_Allgatherv inside).
        // Printing is rank-0 only, but the exchange must be collective across
        // the world communicator. Calling clusterInventory() only on rank 0
        // leaves other ranks out of the collective and causes subsequent
        // MPI collectives (e.g. syncInitStep Allreduce) to mis-match and
        // report MPI_ERR_TRUNCATE.
        const auto &cluster = mpi_ctx->concrete_topology().clusterInventory();
        if (mpi_ctx->rank() == 0)
        {
            InventoryPrinter::printClusterInventory(cluster);
        }

        std::optional<MoEExpertOverlayExecutionPlan> overlay_execution_plan;
        if (config.moe_expert_parallel_plan && config.moe_expert_parallel_plan->isTieredOverlay())
        {
            try
            {
                overlay_execution_plan = resolveMoEExpertOverlayExecutionPlan(
                    config.moe_expert_parallel_plan,
                    MoEExpertOverlayExecutionPlanResolverOptions{
                        .current_world_rank = mpi_ctx->rank(),
                        .world_size = mpi_ctx->world_size(),
                    });
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[Main] failed to resolve MoE expert overlay execution plan: " << e.what());
                mpiShutdown();
                return std::nullopt;
            }

            if (debugEnv().moe_expert_overlay.trace && mpi_ctx->rank() == 0)
            {
                LOG_INFO("[MoEExpertOverlayExecutionPlan]\n"
                         << overlay_execution_plan->diagnostics());
            }
        }

        // --explain-placement: dump the resolved orchestration config on rank 0.
        if (config.explain_placement && mpi_ctx->rank() == 0)
        {
            std::cout << "\n=== Placement Explanation ===\n"
                      << config.toString() << std::endl;
            if (overlay_execution_plan)
            {
                std::cout << "\n=== MoE Expert Overlay Role Plan ===\n"
                          << overlay_execution_plan->diagnostics() << std::endl;
            }
        }

        // Validate model path
        if (config.model_path.empty())
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Error: Model path required (-m)\n\n");
                std::cout << OrchestrationConfigParser::getHelpText() << std::endl;
            }
            if (config.dry_run)
                config.dry_run = false;
            mpiShutdown();
            return std::nullopt;
        }

        // Create and initialize OrchestrationRunner
        auto factory = createOrchestrationRunnerFactory();
        auto runner = factory->createFromOrchestrationConfig(config);

        if (!runner)
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Error: Failed to create orchestration runner");
            }
            if (config.dry_run)
                config.dry_run = false;
            mpiShutdown();
            return std::nullopt;
        }

        if (config.dry_run)
        {
            runDryRunPreflight(config, *runner, mpi_ctx->rank(), std::cout);
            mpiShutdown();
            return std::nullopt;
        }

        if (!runner->initialize())
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Failed to initialize: " << runner->lastError());
            }
            mpiShutdown();
            return std::nullopt;
        }

        // Get tokenizer
        auto tokenizer = runner->tokenizer();
        if (!tokenizer)
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Failed to get tokenizer from runner");
            }
            mpiShutdown();
            return std::nullopt;
        }

        // Apply chat template override
        ChatTemplateResolver::resolve(config.chat_template_override, tokenizer, mpi_ctx->rank());

        // Model-specific chat template override. Applied only when the user has
        // not provided an explicit --chat-template, so user wishes take priority.
        // This lets a model's graph-config builder bundle a community-maintained
        // template that fixes known issues with the GGUF-embedded one.
        if (config.chat_template_override.empty())
        {
            const std::string &architecture = runner->architecture();
            if (!architecture.empty())
            {
                auto config_builder = createGraphConfigBuilder(architecture);
                if (config_builder)
                {
                    auto model_template = config_builder->chatTemplateOverride();
                    if (model_template.has_value() && !model_template->empty())
                    {
                        if (mpi_ctx->rank() == 0)
                        {
                            LOG_DEBUG("Using model-specific chat template override for '"
                                      << architecture << "' ("
                                      << model_template->size() << " bytes)");
                        }
                        tokenizer->setChatTemplate(
                            ChatTemplate::create(*model_template, "", ""));
                    }
                }
            }
        }

        return AppContext{
            .config = config,
            .mpi_ctx = std::move(mpi_ctx),
            .runner = std::move(runner),
            .tokenizer = std::move(tokenizer)};
    }

} // namespace llaminar2
