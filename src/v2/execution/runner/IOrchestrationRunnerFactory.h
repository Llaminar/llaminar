/**
 * @file IOrchestrationRunnerFactory.h
 * @brief Interface for creating orchestration runners
 *
 * Provides a mockable factory interface for creating IOrchestrationRunner
 * instances from various configuration sources:
 * - Command line arguments
 * - YAML configuration files
 * - Direct OrchestrationConfig objects
 *
 * The factory handles dependency injection of:
 * - IOrchestrationConfigParser (CLI/YAML parsing)
 * - IExecutionPlanBuilder (plan generation)
 * - Model loading infrastructure
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "IOrchestrationRunner.h"
#include "../../config/IOrchestrationConfigParser.h"
#include "../mpi_orchestration/IExecutionPlanBuilder.h"
#include <memory>
#include <string>

namespace llaminar2
{

    /**
     * @brief Interface for creating orchestration runners
     *
     * This interface is mockable for unit testing. The concrete implementation
     * (OrchestrationRunnerFactory) creates real runners with all dependencies.
     *
     * Usage patterns:
     *
     * 1. From command line (main entry point):
     *    auto factory = createOrchestrationRunnerFactory();
     *    auto runner = factory->createFromArgs(argc, argv);
     *
     * 2. From config file:
     *    auto runner = factory->createFromConfig("cluster.yaml");
     *
     * 3. From programmatic config (testing):
     *    OrchestrationConfig config;
     *    config.devices.push_back(parseGlobalDeviceAddress("cuda:0"));
     *    auto runner = factory->createFromOrchestrationConfig(config);
     */
    class IOrchestrationRunnerFactory
    {
    public:
        virtual ~IOrchestrationRunnerFactory() = default;

        /**
         * @brief Create runner from command line arguments
         *
         * Parses CLI args to OrchestrationConfig, builds execution plan,
         * and creates the runner. This is the main entry point for the CLI.
         *
         * @param argc Argument count
         * @param argv Argument values
         * @return Runner instance, or nullptr on parse failure
         */
        virtual std::unique_ptr<IOrchestrationRunner> createFromArgs(
            int argc, const char *argv[]) = 0;

        /**
         * @brief Create runner from YAML config file
         *
         * Loads and parses YAML configuration, then creates runner.
         *
         * @param config_path Path to YAML configuration file
         * @return Runner instance, or nullptr on parse failure
         */
        virtual std::unique_ptr<IOrchestrationRunner> createFromConfig(
            const std::string &config_path) = 0;

        /**
         * @brief Create runner from OrchestrationConfig directly
         *
         * Skips parsing and creates runner from pre-built config.
         * Useful for programmatic configuration and testing.
         *
         * @param config Orchestration configuration
         * @return Runner instance, or nullptr on validation failure
         */
        virtual std::unique_ptr<IOrchestrationRunner> createFromOrchestrationConfig(
            const OrchestrationConfig &config) = 0;

        /**
         * @brief Create runner with injected model path
         *
         * Convenience method that creates a simple runner with default settings.
         *
         * @param model_path Path to GGUF model file
         * @param device_spec Device specification (e.g., "cuda:0", "cpu:0")
         * @return Runner instance, or nullptr on failure
         */
        virtual std::unique_ptr<IOrchestrationRunner> createSimple(
            const std::string &model_path,
            const std::string &device_spec = "cpu:0") = 0;
    };

    /**
     * @brief Factory function to create the default factory implementation
     * @return Unique pointer to factory implementation
     */
    std::unique_ptr<IOrchestrationRunnerFactory> createOrchestrationRunnerFactory();

    /**
     * @brief Factory function with injected dependencies (for testing)
     *
     * @param config_parser Config parser (nullptr for default)
     * @param plan_builder Plan builder (nullptr for default)
     * @return Unique pointer to factory implementation
     */
    std::unique_ptr<IOrchestrationRunnerFactory> createOrchestrationRunnerFactory(
        std::unique_ptr<IOrchestrationConfigParser> config_parser,
        std::unique_ptr<IExecutionPlanBuilder> plan_builder);

} // namespace llaminar2
