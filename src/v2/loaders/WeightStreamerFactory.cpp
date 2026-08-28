/**
 * @file WeightStreamerFactory.cpp
 * @brief Implementation of WeightStreamerFactory
 * @author GitHub Copilot
 * @date January 2026
 */

#include "WeightStreamerFactory.h"
#include "../utils/Logger.h"

namespace llaminar2
{

    std::unique_ptr<IWeightStreamer> WeightStreamerFactory::createFromEnv(
        std::shared_ptr<WeightManager> weight_manager,
        int num_layers)
    {
        if (isWeightStreamingEnabled())
        {
            // Streaming mode - validate required parameters
            if (!weight_manager)
            {
                throw std::invalid_argument(
                    "WeightStreamerFactory::createFromEnv: weight_manager is required when "
                    "LLAMINAR_WEIGHT_STREAMING=1");
            }
            if (num_layers <= 0)
            {
                throw std::invalid_argument(
                    "WeightStreamerFactory::createFromEnv: num_layers must be > 0 when "
                    "LLAMINAR_WEIGHT_STREAMING=1, got: " +
                    std::to_string(num_layers));
            }

            // Create config from environment variables
            StreamingConfig config = createStreamingConfigFromEnv();

            LOG_DEBUG("[WeightStreamerFactory] Creating LayerWeightStreamer (streaming mode)");
            LOG_DEBUG("[WeightStreamerFactory]   num_layers=" << num_layers);
            LOG_DEBUG("[WeightStreamerFactory]   gpu_memory_budget="
                      << (config.gpu_memory_budget / (1024 * 1024)) << " MB");
            LOG_DEBUG("[WeightStreamerFactory]   prefetch_depth=" << config.prefetch_depth);

            return std::make_unique<LayerWeightStreamer>(
                std::move(weight_manager), num_layers, config);
        }
        else
        {
            // Resident mode - no streaming needed
            LOG_DEBUG("[WeightStreamerFactory] Creating NullWeightStreamer (resident mode)");
            return std::make_unique<NullWeightStreamer>();
        }
    }

    std::unique_ptr<IWeightStreamer> WeightStreamerFactory::create(
        WeightResidencyMode mode,
        std::shared_ptr<WeightManager> weight_manager,
        int num_layers,
        const StreamingConfig &config)
    {
        switch (mode)
        {
        case WeightResidencyMode::RESIDENT:
            LOG_DEBUG("[WeightStreamerFactory] Creating NullWeightStreamer (RESIDENT mode)");
            return std::make_unique<NullWeightStreamer>();

        case WeightResidencyMode::STREAMING:
        {
            // Validate required parameters for streaming mode
            if (!weight_manager)
            {
                throw std::invalid_argument(
                    "WeightStreamerFactory::create: weight_manager is required for STREAMING mode");
            }
            if (num_layers <= 0)
            {
                throw std::invalid_argument(
                    "WeightStreamerFactory::create: num_layers must be > 0 for STREAMING mode, got: " +
                    std::to_string(num_layers));
            }

            LOG_DEBUG("[WeightStreamerFactory] Creating LayerWeightStreamer (STREAMING mode)");
            LOG_DEBUG("[WeightStreamerFactory]   num_layers=" << num_layers);
            LOG_DEBUG("[WeightStreamerFactory]   gpu_memory_budget="
                      << (config.gpu_memory_budget / (1024 * 1024)) << " MB");

            return std::make_unique<LayerWeightStreamer>(
                std::move(weight_manager), num_layers, config);
        }

        case WeightResidencyMode::UNIFIED:
            // Unified memory mode - let driver handle placement
            // Use NullWeightStreamer since no explicit streaming is needed
            LOG_DEBUG("[WeightStreamerFactory] Creating NullWeightStreamer (UNIFIED mode)");
            return std::make_unique<NullWeightStreamer>();

        default:
            // Should not happen, but provide fallback
            LOG_WARN("[WeightStreamerFactory] Unknown WeightResidencyMode, defaulting to NullWeightStreamer");
            return std::make_unique<NullWeightStreamer>();
        }
    }

} // namespace llaminar2
