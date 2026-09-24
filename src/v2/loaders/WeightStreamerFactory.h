/**
 * @file WeightStreamerFactory.h
 * @brief Factory for creating IWeightStreamer implementations
 * @author GitHub Copilot
 * @date January 2026
 *
 * This file provides a factory for creating the appropriate IWeightStreamer
 * implementation based on configuration, environment variables, or auto-detection.
 *
 * Factory Methods:
 * - createFromEnv(): Uses environment variables (LLAMINAR_WEIGHT_STREAMING)
 * - create(): Uses explicit WeightResidencyMode
 *
 * Decision Logic:
 * 1. RESIDENT mode → NullWeightStreamer (no-op, weights stay on device)
 * 2. STREAMING mode → LayerWeightStreamer (LRU cache, on-demand transfer)
 * 3. UNIFIED mode → NullWeightStreamer (let driver handle placement)
 *
 * @see IWeightStreamer.h for the interface definition
 * @see NullWeightStreamer.h for RESIDENT mode implementation
 * @see LayerWeightStreamer.h for STREAMING mode implementation
 * @see docs/v2/OPTION_B_WEIGHT_STREAMING_DESIGN.md for design details
 */

#pragma once

#include "IWeightStreamer.h"
#include "NullWeightStreamer.h"
#include "LayerWeightStreamer.h"
#include "StreamingConfigFromEnv.h"
#include "WeightManager.h"

#include <memory>
#include <stdexcept>

namespace llaminar2
{

    /**
     * @brief Factory for creating IWeightStreamer implementations
     *
     * Decides which streamer to use based on:
     * 1. Environment variables (LLAMINAR_WEIGHT_STREAMING)
     * 2. Explicit configuration (WeightResidencyMode)
     * 3. Auto-detection (model size vs available VRAM)
     *
     * Usage Examples:
     *
     *   // From environment variables:
     *   auto streamer = WeightStreamerFactory::createFromEnv(weight_mgr, num_layers);
     *
     *   // Explicit mode:
     *   auto streamer = WeightStreamerFactory::create(
     *       WeightResidencyMode::STREAMING, weight_mgr, num_layers);
     *
     *   // Auto-detect mode:
     *   auto streamer = WeightStreamerFactory::create(mode, weight_mgr, num_layers);
     */
    class WeightStreamerFactory
    {
    public:
        /**
         * @brief Create streamer from environment variables
         *
         * If LLAMINAR_WEIGHT_STREAMING=1: Creates LayerWeightStreamer
         * Otherwise: Creates NullWeightStreamer
         *
         * The StreamingConfig is populated from environment variables:
         * - LLAMINAR_STREAM_MEMORY_MB → gpu_memory_budget
         * - LLAMINAR_STREAM_PREFETCH_DEPTH → prefetch_depth
         * - LLAMINAR_STREAM_EVICTION_POLICY → eviction_policy (lru/fifo/none)
         * - LLAMINAR_STREAM_VERBOSE → log_transfer_stats
         *
         * @param weight_manager WeightManager for layer weight access (required for streaming)
         * @param num_layers Number of model layers (required for streaming)
         * @return Unique pointer to IWeightStreamer implementation
         *
         * @throws std::invalid_argument If streaming enabled but weight_manager is null
         * @throws std::invalid_argument If streaming enabled but num_layers <= 0
         */
        static std::unique_ptr<IWeightStreamer> createFromEnv(
            std::shared_ptr<WeightManager> weight_manager,
            int num_layers);

        /**
         * @brief Create streamer from explicit residency mode
         *
         * @param mode RESIDENT, STREAMING, or UNIFIED
         * @param weight_manager WeightManager (required for STREAMING mode)
         * @param num_layers Number of layers (required for STREAMING mode)
         * @param config Optional streaming configuration (only used for STREAMING mode)
         * @return Unique pointer to IWeightStreamer implementation
         *
         * @throws std::invalid_argument If STREAMING mode but weight_manager is null
         * @throws std::invalid_argument If STREAMING mode but num_layers <= 0
         */
        static std::unique_ptr<IWeightStreamer> create(
            WeightResidencyMode mode,
            std::shared_ptr<WeightManager> weight_manager = nullptr,
            int num_layers = 0,
            const StreamingConfig &config = StreamingConfig{});

    private:
        // Private constructor - static factory class
        WeightStreamerFactory() = delete;
    };

} // namespace llaminar2
