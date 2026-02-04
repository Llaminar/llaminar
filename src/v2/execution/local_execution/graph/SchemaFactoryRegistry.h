/**
 * @file SchemaFactoryRegistry.h
 * @brief Registry for architecture-specific schema factories
 * @author David Sanftenberg
 * @date February 2026
 *
 * This file provides a model-agnostic way to obtain schema factories
 * based on architecture string. This allows orchestrators and factories
 * to remain model-agnostic while still accessing architecture-specific
 * configurations like weight sharding patterns.
 *
 * Usage:
 *   std::string arch = model_ctx->architecture();  // e.g., "qwen2"
 *   auto factory = SchemaFactoryRegistry::getFactory(arch);
 *   WeightShardingConfig config = factory->getWeightShardingConfig();
 */

#pragma once

#include "GraphSchema.h"
#include <memory>
#include <stdexcept>
#include <string>

namespace llaminar2
{

    /**
     * @brief Registry for model architecture schema factories
     *
     * Provides model-agnostic access to architecture-specific schema factories.
     * Orchestrators use this to avoid hardcoding model-specific factories.
     *
     * Currently supported architectures:
     * - qwen2: Qwen2SchemaFactory
     *
     * Future architectures can be added by:
     * 1. Creating a new FooSchemaFactory : public ISchemaFactory
     * 2. Adding a case in getFactory() for the architecture name
     */
    class SchemaFactoryRegistry
    {
    public:
        /**
         * @brief Get schema factory for a given architecture
         *
         * @param architecture Model architecture string (e.g., "qwen2")
         * @return Unique pointer to the schema factory for this architecture
         * @throws std::runtime_error if architecture is not supported
         *
         * Example:
         *   auto factory = SchemaFactoryRegistry::getFactory("qwen2");
         *   WeightShardingConfig cfg = factory->getWeightShardingConfig();
         */
        static std::unique_ptr<ISchemaFactory> getFactory(const std::string &architecture);

        /**
         * @brief Get weight sharding config for a given architecture
         *
         * Convenience method that creates a temporary factory and extracts
         * the weight sharding configuration. Useful when only the sharding
         * config is needed without the full factory.
         *
         * @param architecture Model architecture string (e.g., "qwen2")
         * @return WeightShardingConfig for this architecture
         * @throws std::runtime_error if architecture is not supported
         *
         * Example:
         *   auto cfg = SchemaFactoryRegistry::getWeightShardingConfig("qwen2");
         */
        static WeightShardingConfig getWeightShardingConfig(const std::string &architecture);

        /**
         * @brief Check if an architecture is supported
         *
         * @param architecture Model architecture string
         * @return true if the architecture has a registered factory
         */
        static bool isSupported(const std::string &architecture);

        /**
         * @brief Get list of supported architecture names
         *
         * @return Vector of supported architecture strings
         */
        static std::vector<std::string> supportedArchitectures();
    };

} // namespace llaminar2
