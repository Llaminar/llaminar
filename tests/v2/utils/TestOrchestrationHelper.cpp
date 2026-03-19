/**
 * @file TestOrchestrationHelper.cpp
 * @brief Implementation of TestOrchestrationHelper
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "TestOrchestrationHelper.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include "backends/GlobalDeviceAddress.h"
#include <stdexcept>

namespace llaminar2::test
{

    // =========================================================================
    // Simple Runner Creation
    // =========================================================================

    std::unique_ptr<IOrchestrationRunner> TestOrchestrationHelper::createSimple(
        const std::string &model_path,
        DeviceId device,
        int max_seq_len)
    {
        auto config = buildSimpleConfig(model_path, device, max_seq_len);
        return create(config);
    }

    // =========================================================================
    // Full Config Runner Creation
    // =========================================================================

    std::unique_ptr<IOrchestrationRunner> TestOrchestrationHelper::create(
        const OrchestrationConfig &config)
    {
        auto factory = createOrchestrationRunnerFactory();
        return factory->createFromOrchestrationConfig(config);
    }

    // =========================================================================
    // Pipeline Parallel (PP) Configuration
    // =========================================================================

    std::unique_ptr<IOrchestrationRunner> TestOrchestrationHelper::forPPStage(
        const std::string &model_path,
        int stage_idx,
        int total_stages,
        DeviceId device,
        int total_layers)
    {
        auto config = buildPPStageConfig(model_path, stage_idx, total_stages, device, total_layers);
        return create(config);
    }

    // =========================================================================
    // Tensor Parallel (TP) Configuration
    // =========================================================================

    std::unique_ptr<IOrchestrationRunner> TestOrchestrationHelper::forTPShard(
        const std::string &model_path,
        int shard_idx,
        int total_shards,
        DeviceId device)
    {
        auto config = buildTPShardConfig(model_path, shard_idx, total_shards, device);
        return create(config);
    }

    // =========================================================================
    // Configuration Builders
    // =========================================================================

    OrchestrationConfig TestOrchestrationHelper::buildSimpleConfig(
        const std::string &model_path,
        DeviceId device,
        int max_seq_len)
    {
        OrchestrationConfig config = OrchestrationConfig::defaults();

        // Model configuration
        config.model_path = model_path;
        config.max_seq_len = max_seq_len;

        // Single device configuration via device map (rank 0 → chosen device)
        config.device_mode = DeviceAssignmentMode::EXPLICIT;
        config.device_map.emplace_back(0, toGlobalAddress(device));

        // No parallelism
        config.tp_degree = 1;
        config.pp_degree = 1;

        // Deterministic for testing
        config.deterministic = true;

        return config;
    }

    OrchestrationConfig TestOrchestrationHelper::buildPPStageConfig(
        const std::string &model_path,
        int stage_idx,
        int total_stages,
        DeviceId device,
        int total_layers)
    {
        if (stage_idx < 0 || stage_idx >= total_stages)
        {
            throw std::invalid_argument(
                "Invalid stage_idx " + std::to_string(stage_idx) +
                " for total_stages " + std::to_string(total_stages));
        }

        if (total_stages < 1)
        {
            throw std::invalid_argument(
                "total_stages must be >= 1, got " + std::to_string(total_stages));
        }

        OrchestrationConfig config = OrchestrationConfig::defaults();

        // Model configuration
        config.model_path = model_path;

        // Device configuration
        config.device_mode = DeviceAssignmentMode::EXPLICIT;
        config.device_for_this_rank = toGlobalAddress(device, stage_idx);

        // Pipeline parallelism configuration
        config.pp_degree = total_stages;
        config.pp_split = PPSplitMode::EQUAL;

        // No tensor parallelism within stage (can be combined separately)
        config.tp_degree = 1;

        // Deterministic for testing
        config.deterministic = true;

        // If total_layers specified, we could set up explicit layer ranges
        // but PPSplitMode::EQUAL handles this automatically
        (void)total_layers; // Reserved for future use

        return config;
    }

    OrchestrationConfig TestOrchestrationHelper::buildTPShardConfig(
        const std::string &model_path,
        int shard_idx,
        int total_shards,
        DeviceId device)
    {
        if (shard_idx < 0 || shard_idx >= total_shards)
        {
            throw std::invalid_argument(
                "Invalid shard_idx " + std::to_string(shard_idx) +
                " for total_shards " + std::to_string(total_shards));
        }

        if (total_shards < 1)
        {
            throw std::invalid_argument(
                "total_shards must be >= 1, got " + std::to_string(total_shards));
        }

        OrchestrationConfig config = OrchestrationConfig::defaults();

        // Model configuration
        config.model_path = model_path;

        // Device configuration
        config.device_mode = DeviceAssignmentMode::EXPLICIT;
        config.device_for_this_rank = toGlobalAddress(device, shard_idx);

        // Tensor parallelism configuration
        config.tp_degree = total_shards;
        config.tp_scope = TPScope::GLOBAL; // Cross-rank TP for explicit shard testing

        // No pipeline parallelism
        config.pp_degree = 1;

        // Enable weight sharding
        config.shard_weights = true;

        // Deterministic for testing
        config.deterministic = true;

        return config;
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    GlobalDeviceAddress TestOrchestrationHelper::toGlobalAddress(DeviceId device, int rank)
    {
        // GlobalDeviceAddress uses hostname for multi-node, not rank
        // For test purposes, we use "localhost" and encode rank in NUMA node
        // This maps DeviceId to a GlobalDeviceAddress that the factory can use
        return GlobalDeviceAddress::fromLocalDeviceId(device, "localhost", rank);
    }

} // namespace llaminar2::test
