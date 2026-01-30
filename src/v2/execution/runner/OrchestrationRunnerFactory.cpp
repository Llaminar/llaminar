/**
 * @file OrchestrationRunnerFactory.cpp
 * @brief Implementation of OrchestrationRunnerFactory
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "IOrchestrationRunnerFactory.h"
#include "OrchestrationRunner.h"
#include "../../config/OrchestrationConfigParser.h"
#include "../mpi_orchestration/ExecutionPlanBuilder.h"
#include "../../utils/Logger.h"
#include <fstream>

namespace llaminar2
{

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

                return createFromOrchestrationConfig(config);
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

                return createFromOrchestrationConfig(config);
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to parse config file: " << e.what());
                return nullptr;
            }
        }

        std::unique_ptr<IOrchestrationRunner> createFromOrchestrationConfig(
            const OrchestrationConfig &config) override
        {
            // Create a copy of plan builder for the runner
            // (each runner needs its own instance)
            auto runner_plan_builder = createExecutionPlanBuilder();

            auto runner = std::make_unique<OrchestrationRunner>(
                config,
                std::move(runner_plan_builder));

            return runner;
        }

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

            return createFromOrchestrationConfig(config);
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
