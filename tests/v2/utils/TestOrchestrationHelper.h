/**
 * @file TestOrchestrationHelper.h
 * @brief Simplified orchestration helper for test infrastructure
 *
 * Provides convenience methods for creating OrchestrationRunner instances
 * in tests without the complexity of full OrchestrationConfig setup.
 *
 * Usage:
 *   #include "utils/TestOrchestrationHelper.h"
 *   using namespace llaminar2::test;
 *
 *   // Simple single-device runner
 *   auto runner = TestOrchestrationHelper::createSimple("model.gguf", DeviceId::cpu());
 *
 *   // Full config
 *   OrchestrationConfig config;
 *   config.model_path = "model.gguf";
 *   auto runner = TestOrchestrationHelper::create(config);
 *
 *   // PP stage runner
 *   auto runner = TestOrchestrationHelper::forPPStage("model.gguf", 0, 2, DeviceId::cuda(0));
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "backends/DeviceId.h"
#include "config/OrchestrationConfig.h"
#include "execution/runner/IOrchestrationRunner.h"
#include <memory>
#include <string>
#include <optional>

namespace llaminar2::test
{

    /**
     * @brief Simplified orchestration helper for tests
     *
     * Wraps OrchestrationRunnerFactory with test-friendly convenience methods
     * that reduce boilerplate for common test scenarios.
     */
    class TestOrchestrationHelper
    {
    public:
        // =========================================================================
        // Simple Runner Creation
        // =========================================================================

        /**
         * @brief Create a simple single-device runner
         *
         * Creates an OrchestrationRunner with minimal configuration:
         * - Single device execution
         * - Default sequence length (2048)
         * - No tensor or pipeline parallelism
         *
         * @param model_path Path to GGUF model file
         * @param device Target device (default: CPU)
         * @param max_seq_len Maximum sequence length (default: 2048)
         * @return Runner instance, or nullptr on failure
         */
        static std::unique_ptr<IOrchestrationRunner> createSimple(
            const std::string &model_path,
            DeviceId device = DeviceId::cpu(),
            int max_seq_len = 2048);

        // =========================================================================
        // Full Config Runner Creation
        // =========================================================================

        /**
         * @brief Create runner from full OrchestrationConfig
         *
         * Passes through to OrchestrationRunnerFactory::createFromOrchestrationConfig().
         * Use this when you need fine-grained control over configuration.
         *
         * @param config Complete orchestration configuration
         * @return Runner instance, or nullptr on validation failure
         */
        static std::unique_ptr<IOrchestrationRunner> create(
            const OrchestrationConfig &config);

        // =========================================================================
        // Pipeline Parallel (PP) Configuration
        // =========================================================================

        /**
         * @brief Create runner configured for a specific PP stage
         *
         * Configures the runner for pipeline parallel execution where this
         * runner handles a specific subset of layers.
         *
         * @param model_path Path to GGUF model file
         * @param stage_idx PP stage index (0-based)
         * @param total_stages Total number of PP stages
         * @param device Target device for this stage
         * @param total_layers Total layers in model (default: 0 = auto-detect)
         * @return Runner instance, or nullptr on failure
         *
         * @note Stage 0 includes embedding, final stage includes LM head.
         *       Layer distribution is approximately equal across stages.
         */
        static std::unique_ptr<IOrchestrationRunner> forPPStage(
            const std::string &model_path,
            int stage_idx,
            int total_stages,
            DeviceId device,
            int total_layers = 0);

        // =========================================================================
        // Tensor Parallel (TP) Configuration
        // =========================================================================

        /**
         * @brief Create runner configured for a specific TP shard
         *
         * Configures the runner for tensor parallel execution where weights
         * are sharded across devices. This is typically used within a single
         * node with multiple GPUs.
         *
         * @param model_path Path to GGUF model file
         * @param shard_idx TP shard index (0-based)
         * @param total_shards Total number of TP shards
         * @param device Target device for this shard
         * @return Runner instance, or nullptr on failure
         *
         * @note For LOCAL TP, use createSimple() with tp_devices in config instead.
         *       This method is for explicit shard control in testing.
         */
        static std::unique_ptr<IOrchestrationRunner> forTPShard(
            const std::string &model_path,
            int shard_idx,
            int total_shards,
            DeviceId device);

        // =========================================================================
        // Configuration Builders (for advanced use)
        // =========================================================================

        /**
         * @brief Build OrchestrationConfig for single-device execution
         *
         * @param model_path Path to GGUF model file
         * @param device Target device
         * @param max_seq_len Maximum sequence length
         * @return Configured OrchestrationConfig
         */
        static OrchestrationConfig buildSimpleConfig(
            const std::string &model_path,
            DeviceId device,
            int max_seq_len = 2048);

        /**
         * @brief Build OrchestrationConfig for PP stage
         *
         * @param model_path Path to GGUF model file
         * @param stage_idx PP stage index
         * @param total_stages Total PP stages
         * @param device Target device
         * @param total_layers Total model layers (0 = auto)
         * @return Configured OrchestrationConfig
         */
        static OrchestrationConfig buildPPStageConfig(
            const std::string &model_path,
            int stage_idx,
            int total_stages,
            DeviceId device,
            int total_layers = 0);

        /**
         * @brief Build OrchestrationConfig for TP shard
         *
         * @param model_path Path to GGUF model file
         * @param shard_idx TP shard index
         * @param total_shards Total TP shards
         * @param device Target device
         * @return Configured OrchestrationConfig
         */
        static OrchestrationConfig buildTPShardConfig(
            const std::string &model_path,
            int shard_idx,
            int total_shards,
            DeviceId device);

    private:
        /**
         * @brief Convert DeviceId to GlobalDeviceAddress
         *
         * @param device DeviceId to convert
         * @param rank MPI rank (default: 0)
         * @return GlobalDeviceAddress for the device
         */
        static GlobalDeviceAddress toGlobalAddress(DeviceId device, int rank = 0);
    };

} // namespace llaminar2::test
