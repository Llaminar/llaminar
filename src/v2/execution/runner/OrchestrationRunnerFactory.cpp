/**
 * @file OrchestrationRunnerFactory.cpp
 * @brief Implementation of OrchestrationRunnerFactory
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "IOrchestrationRunnerFactory.h"
#include "OrchestrationRunner.h"
#include "NamedDomainGlobalRunner.h"
#include "../../config/OrchestrationConfigParser.h"
#include "../../config/ParallelismTreeParser.h"
#include "../mpi_orchestration/ExecutionPlanBuilder.h"
#include "../parallelism_tree/TreeToRunnerCompiler.h"
#include "../../utils/Logger.h"
#include "../../utils/MPIContext.h"
#include <fstream>

namespace llaminar2
{
    RunnerModelAuthorityScope resolveRunnerModelAuthorityScope(
        const OrchestrationConfig &config)
    {
        if (config.topology_tree || NamedDomainGlobalRunner::shouldUse(config))
            return RunnerModelAuthorityScope::MultiRankSet;
        return RunnerModelAuthorityScope::RankLocal;
    }

    /**
     * @brief Concrete implementation of IOrchestrationRunnerFactory
     */
    class OrchestrationRunnerFactory : public IOrchestrationRunnerFactory
    {
    public:
        /**
         * @brief Construct with default dependencies
         */
        OrchestrationRunnerFactory()
            : config_parser_(createOrchestrationConfigParser()), plan_builder_(createExecutionPlanBuilder())
        {
        }

        /**
         * @brief Construct with injected dependencies
         */
        OrchestrationRunnerFactory(
            std::unique_ptr<IOrchestrationConfigParser> config_parser,
            std::unique_ptr<IExecutionPlanBuilder> plan_builder)
            : config_parser_(std::move(config_parser)), plan_builder_(std::move(plan_builder))
        {
            // Use defaults if not provided
            if (!config_parser_)
            {
                config_parser_ = createOrchestrationConfigParser();
            }
            if (!plan_builder_)
            {
                plan_builder_ = createExecutionPlanBuilder();
            }
        }

        std::unique_ptr<IOrchestrationRunner> createFromArgs(
            int argc, const char *argv[]) override
        {
            if (argc < 1 || !argv)
            {
                LOG_ERROR("Invalid arguments to createFromArgs");
                return nullptr;
            }

            try
            {
                // Convert const char** to char** for parser
                std::vector<char *> mutable_argv(argc);
                for (int i = 0; i < argc; ++i)
                {
                    mutable_argv[i] = const_cast<char *>(argv[i]);
                }

                // Parse CLI arguments
                OrchestrationConfig config = config_parser_->parseArgs(
                    argc, mutable_argv.data());

                // Validate configuration
                auto errors = config.validate();
                if (!errors.empty())
                {
                    LOG_ERROR("Configuration validation failed:");
                    for (const auto &e : errors)
                    {
                        LOG_ERROR("  - " << e);
                    }
                    return nullptr;
                }

                return createFromOrchestrationConfig(std::move(config));
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to parse arguments: " << e.what());
                return nullptr;
            }
        }

        std::unique_ptr<IOrchestrationRunner> createFromConfig(
            const std::string &config_path) override
        {
            // Check file exists
            std::ifstream file(config_path);
            if (!file.good())
            {
                LOG_ERROR("Config file not found: " << config_path);
                return nullptr;
            }
            file.close();

            try
            {
                OrchestrationConfig config = config_parser_->parseYamlFile(config_path);
                config.config_file_path = config_path;

                // Validate configuration
                auto errors = config.validate();
                if (!errors.empty())
                {
                    LOG_ERROR("Configuration validation failed:");
                    for (const auto &e : errors)
                    {
                        LOG_ERROR("  - " << e);
                    }
                    return nullptr;
                }

                return createFromOrchestrationConfig(std::move(config));
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to parse config file: " << e.what());
                return nullptr;
            }
        }

        std::unique_ptr<IOrchestrationRunner> createFromOrchestrationConfig(
            OrchestrationConfig config) override
        {
            return createFromOrchestrationConfigImpl(
                std::move(config),
                nullptr,
                std::nullopt,
                nullptr,
                {});
        }

        std::unique_ptr<IOrchestrationRunner> createFromOrchestrationConfig(
            OrchestrationConfig config,
            std::shared_ptr<ModelContext> model_context) override
        {
            if (!model_context)
            {
                LOG_ERROR(
                    "Preloaded ModelContext runner construction requires a "
                    "non-null model authority");
                return nullptr;
            }
            return createFromOrchestrationConfigImpl(
                std::move(config),
                std::move(model_context),
                std::nullopt,
                nullptr,
                {});
        }

        std::unique_ptr<IOrchestrationRunner> createFromOrchestrationConfig(
            OrchestrationConfig config,
            ModelContextReuseContract reuse_contract) override
        {
            if (!reuse_contract.context)
            {
                LOG_ERROR(
                    "Retained ModelContext runner construction requires a "
                    "non-null model authority");
                return nullptr;
            }
            auto model_context = std::move(reuse_contract.context);
            auto prepared_weight_plan =
                std::move(reuse_contract.prepared_weight_plan);
            auto prepared_routed_weight_plan =
                std::move(reuse_contract.prepared_routed_weight_plan);
            auto routed_weight_authority_identity =
                std::move(reuse_contract.routed_weight_authority_identity);
            return createFromOrchestrationConfigImpl(
                std::move(config),
                std::move(model_context),
                std::move(prepared_weight_plan),
                std::move(prepared_routed_weight_plan),
                std::move(routed_weight_authority_identity));
        }

    private:
        /**
         * @brief Normalize configuration and construct the selected runner.
         *
         * A retained ModelContext is a rank-local OrchestrationRunner contract.
         * Legacy cross-rank TP/PP still constructs one such runner per MPI rank;
         * named-domain and topology-tree runners compose multiple rank-specific
         * authorities and reject this overload until their builders accept a
         * typed context set.
         */
        std::unique_ptr<IOrchestrationRunner> createFromOrchestrationConfigImpl(
            OrchestrationConfig config,
            std::shared_ptr<ModelContext> model_context,
            std::optional<RankExecutionPlan> prepared_weight_plan,
            std::shared_ptr<const MoERoutedExpertPlacementPlan>
                prepared_routed_weight_plan,
            std::string routed_weight_authority_identity)
        {
            auto normalize_errors = normalizeMoERoutedExpertPlacementDomains(config);
            if (!normalize_errors.empty())
            {
                LOG_ERROR("MoE expert overlay domain normalization failed:");
                for (const auto &error : normalize_errors)
                {
                    LOG_ERROR("  - " << error);
                }
                return nullptr;
            }

            if (model_context &&
                resolveRunnerModelAuthorityScope(config) ==
                    RunnerModelAuthorityScope::MultiRankSet)
            {
                LOG_ERROR(
                    "A rank-local ModelContext cannot construct a runner that "
                    "owns a multi-rank model-authority set");
                return nullptr;
            }

            // ================================================================
            // Handle topology tree if present (Phase 8: Global PP integration)
            // ================================================================
            if (config.topology_tree)
            {
                LOG_DEBUG("Using ParallelismTree topology for runner creation");
                
                // Get MPI context for world size and rank
                auto mpi_ctx = MPIContextFactory::global();
                
                // Build compile context
                TreeToRunnerCompiler::CompileContext compile_ctx;
                compile_ctx.my_rank = mpi_ctx->rank();
                compile_ctx.world_size = mpi_ctx->world_size();
                compile_ctx.max_seq_len = static_cast<size_t>(config.max_seq_len);
                compile_ctx.batch_size = config.batch_size;
                // Note: hidden_dim and vocab_size will be set from model once loaded
                compile_ctx.hidden_dim = 896;  // Qwen2.5 default
                compile_ctx.vocab_size = 151936;  // Qwen2.5 default
                
                // For now, log tree structure and fall back to standard path
                // Full tree-to-runner compilation requires model loading first
                LOG_DEBUG("Topology tree structure:\n" << config.topology_tree->toString());
                LOG_DEBUG("Note: Full tree compilation requires loaded model context");
                LOG_DEBUG("Falling back to standard OrchestrationRunner path");
                
                // The actual tree compilation would be:
                // auto runner = TreeToRunnerCompiler::compile(*config.topology_tree, compile_ctx);
                // But we need model context first, so fall through to standard path
            }

            // ================================================================
            // Phase 5: Named-domain global pipeline runner
            // ================================================================
            // When the config has named domains with PP stages that span
            // multiple MPI ranks, use NamedDomainGlobalRunner.  This supports
            // scope=node_local, scope=global, and AUTO domains whose device
            // list spans multiple hostnames.
            if (NamedDomainGlobalRunner::shouldUse(config))
            {
                LOG_DEBUG("Named-domain global PP configuration detected — using NamedDomainGlobalRunner");
                auto runner_plan_builder = createExecutionPlanBuilder();
                return std::make_unique<NamedDomainGlobalRunner>(
                    std::move(config),
                    std::move(runner_plan_builder));
            }

            // Global orchestration detection (legacy path — simple --pp-degree mode)
            {
                auto mpi_ctx = MPIContextFactory::global();
                int world_size = mpi_ctx->world_size();
                bool needs_global = (world_size > 1 && config.pp_degree > 1) ||
                                    config.tp_scope == TPScope::GLOBAL ||
                                    config.tp_scope == TPScope::NODE_LOCAL;

                if (needs_global)
                {
                    LOG_DEBUG("Global orchestration conditions detected ("
                             << "world_size=" << world_size
                             << ", pp_degree=" << config.pp_degree
                             << ", tp_scope=" << tpScopeToString(config.tp_scope) << ")");
                    LOG_DEBUG(
                        "Selecting one ordinary OrchestrationRunner per MPI "
                        "rank; its RankExecutionPlan owns the cross-rank "
                        "collective edges");
                    if (model_context)
                    {
                        LOG_DEBUG(
                            "Using the calling MPI rank's ModelContext in the "
                            "rank-local OrchestrationRunner");
                    }
                }
            }

            // Standard path: Create OrchestrationRunner with ExecutionPlanBuilder
            // Create a copy of plan builder for the runner
            // (each runner needs its own instance)
            auto runner_plan_builder = createExecutionPlanBuilder();

            if (prepared_weight_plan)
            {
                return std::make_unique<OrchestrationRunner>(
                    std::move(config),
                    std::move(runner_plan_builder),
                    ModelContextReuseContract{
                        .context = std::move(model_context),
                        .prepared_weight_plan =
                            std::move(*prepared_weight_plan),
                        .prepared_routed_weight_plan =
                            std::move(prepared_routed_weight_plan),
                        .routed_weight_authority_identity =
                            std::move(routed_weight_authority_identity),
                    });
            }
            if (model_context)
            {
                return std::make_unique<OrchestrationRunner>(
                    std::move(config),
                    std::move(runner_plan_builder),
                    std::move(model_context));
            }
            return std::make_unique<OrchestrationRunner>(
                std::move(config),
                std::move(runner_plan_builder));
        }

    public:

        std::unique_ptr<IOrchestrationRunner> createSimple(
            const std::string &model_path,
            const std::string &device_spec) override
        {
            OrchestrationConfig config = OrchestrationConfig::defaults();

            // Parse device spec
            auto device = GlobalDeviceAddress::tryParse(device_spec);
            if (!device.has_value())
            {
                LOG_ERROR("Invalid device specification: " << device_spec);
                return nullptr;
            }

            config.device_for_this_rank = *device;
            config.tp_degree = 1;
            config.pp_degree = 1;

            // Note: model_path would typically be stored in config
            // For now, the runner will need to be configured separately

            return createFromOrchestrationConfig(std::move(config));
        }

    private:
        std::unique_ptr<IOrchestrationConfigParser> config_parser_;
        std::unique_ptr<IExecutionPlanBuilder> plan_builder_;
    };

    // =========================================================================
    // Factory Functions
    // =========================================================================

    std::unique_ptr<IOrchestrationRunnerFactory> createOrchestrationRunnerFactory()
    {
        return std::make_unique<OrchestrationRunnerFactory>();
    }

    std::unique_ptr<IOrchestrationRunnerFactory> createOrchestrationRunnerFactory(
        std::unique_ptr<IOrchestrationConfigParser> config_parser,
        std::unique_ptr<IExecutionPlanBuilder> plan_builder)
    {
        return std::make_unique<OrchestrationRunnerFactory>(
            std::move(config_parser),
            std::move(plan_builder));
    }

} // namespace llaminar2
